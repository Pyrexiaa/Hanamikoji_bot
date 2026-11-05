"""Mutable game-state model and state transitions."""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Dict, Generator, List, Optional, Sequence, Set, Tuple

from .constants import CHANCE_PLAYER
from .deck import Deck
from .enums import OFFER_RESPONSES, ActionType
from .legal_actions import iter_offer_response_actions, iter_player_legal_actions
from .render import format_game_state_one_line, pretty_print_game_state, snapshot_game_state
from .events import (
    _freeze_favors,
    CompetitionChoiceEvent,
    DrawCardEvent,
    GiftChoiceEvent,
    OpponentCompetitionEvent,
    OpponentGiftEvent,
    OpponentTradeOffEvent,
    OwnCompetitionEvent,
    OwnGiftEvent,
    OwnTradeOffEvent,
    ObservationEvent,
    OpponentDiscardEvent,
    OpponentReserveEvent,
    OpponentSplitEvent,
    OwnDiscardEvent,
    OwnReserveEvent,
    OwnSplitEvent,
    RoundEndEvent,
    RoundStartEvent,
    SplitChoiceEvent,
)
from .info_state import RoundInfoState
from .models import Action, Card, GameConfig


@dataclass(frozen=True)
class RoundResolutionPreview:
    """Pure end-of-round resolution result without mutating state."""

    p0_reserved_revealed: Tuple[object, ...]
    p1_reserved_revealed: Tuple[object, ...]
    final_favors: Tuple[Tuple[object, int], ...]
    winner: Optional[int]


@dataclass(frozen=True)
class GameStateKey:
    """Hashable value-based snapshot of a GameState. Use GameState.to_key()."""

    phase: str
    current_player: int
    turn_count: int
    round_count: int
    round_starting_player: Optional[int]
    pending_draw_player: Optional[int]
    winner: Optional[int]
    deck: frozenset
    favors: frozenset
    # (offerer, offer type, offered suits) or None
    pending_offer: Optional[Tuple[int, ActionType, Tuple[object, ...]]]
    p0_hand: tuple
    p0_reserved: tuple
    p0_collected: tuple
    p0_discarded_revealed: tuple
    p0_discarded_hidden: tuple
    p0_used_actions: frozenset
    p1_hand: tuple
    p1_reserved: tuple
    p1_collected: tuple
    p1_discarded_revealed: tuple
    p1_discarded_hidden: tuple
    p1_used_actions: frozenset


@dataclass(frozen=True)
class _UndoRecord:
    """Compact pre-action snapshot plus per-action inverse payload."""

    action_type: ActionType
    pre_phase: str
    pre_current_player: int
    pre_turn_count: int
    pre_pending_offer: Optional[Tuple[int, ActionType, Tuple[Card, ...]]]
    pre_pending_draw_player: Optional[int]
    pre_round_starting_player: Optional[int]
    removed_cards: Tuple[Tuple[int, int, Card], ...] = ()
    added_used_action: Optional[Tuple[int, ActionType]] = None
    events_appended_p0: int = 0
    events_appended_p1: int = 0
    dealt_cards: Tuple[Card, ...] = ()
    pre_hand_p0: Tuple[Card, ...] = ()
    pre_hand_p1: Tuple[Card, ...] = ()
    draw_player: Optional[int] = None


class PlayerState:
    """Tracks the state of a single player."""

    def __init__(self, player_id: int, suits_enum: type):
        self.player_id = player_id
        self._suits_enum = suits_enum
        self.hand: List[Card] = []
        self.reserved_cards: List[Card] = []
        self.collected_cards: List[Card] = []
        self.discarded_revealed: List[Card] = []
        self.discarded_hidden: List[Card] = []
        self.used_actions: Set[ActionType] = set()

    def reset_for_round(self):
        self.hand = []
        self.reserved_cards = []
        self.collected_cards = []
        self.discarded_revealed = []
        self.discarded_hidden = []
        self.used_actions = set()

    def get_suit_count(self, include_reserved: bool = True) -> dict:
        counts = {suit: 0 for suit in self._suits_enum}
        for card in self.collected_cards:
            counts[card.suit] += 1
        if include_reserved:
            for card in self.reserved_cards:
                counts[card.suit] += 1
        return counts


class GameState:
    """Manages the overall game state, parameterized by GameConfig."""

    def __init__(self, config: GameConfig):
        self.config = config
        # The remaining cards in the current round. This is a multiset (counts by suit),
        # not an ordered list.
        self.deck = Deck({})
        self.players: List[PlayerState] = [
            PlayerState(0, config.suits),
            PlayerState(1, config.suits),
        ]
        self.favors = {suit: 0 for suit in config.suits}
        # Internal current-player storage. Use the `current_player` property for phase-aware access.
        self._current_player: int = 0
        self.turn_count = 0
        self.round_count = 0
        self.winner: Optional[int] = None
        # An offer awaiting the opponent's response within the same turn:
        # (offerer_id, SPLIT | GIFT | COMPETITION, offered cards in canonical
        # order; a COMPETITION's pairs are cards[0:2] and cards[2:4]).
        self.pending_offer: Optional[Tuple[int, ActionType, Tuple[Card, ...]]] = None
        self.round_starting_player: Optional[int] = None
        self.phase: str = "choose_start"
        self.pending_draw_player: Optional[int] = None
        self._undo_stack: List[_UndoRecord] = []
        self._favors_updated_this_round = False

        # info_state_history[player_id][round_index] -> list of ObservationEvent
        self.info_state_history: List[List[List[ObservationEvent]]] = [[], []]

        # Round 1 begins immediately on construction.
        self._begin_round()

    @property
    def pending_split(self) -> Optional[Tuple[int, Tuple[Card, Card]]]:
        """A pending SPLIT as (offerer_id, (card0, card1)), else None.

        Only covers SPLIT; `pending_offer` covers every offer type.
        """
        if self.pending_offer is None or self.pending_offer[1] != ActionType.SPLIT:
            return None
        offerer_id, _, cards = self.pending_offer
        return offerer_id, (cards[0], cards[1])

    @property
    def current_player(self) -> Optional[int]:
        """Return the acting player id, or `None` when no actor exists.

        - Returns `CHANCE_PLAYER` (-1) during chance phases.
        - Returns `0` or `1` during normal play.
        - Returns `None` when `phase == "round_complete"`.

        This avoids exposing a stale "current player" after the round has already ended.
        """
        if self.phase == "round_complete":
            return None
        return self._current_player

    @current_player.setter
    def current_player(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError(f"current_player must be an int, got {type(value).__name__}")
        if value not in (CHANCE_PLAYER, 0, 1):
            raise ValueError(f"current_player must be CHANCE_PLAYER, 0, or 1 (got {value}).")
        self._current_player = value

    def _remaining_total(self) -> int:
        return self.deck.total()

    def _initial_deal_cards_count(self) -> int:
        return 2 * self.config.initial_hand_size

    def _begin_round(self) -> None:
        if self.winner is not None:
            raise RuntimeError("Cannot begin a new round after the game has a winner.")
        if self.round_count > 0 and self.phase != "round_complete":
            raise RuntimeError(
                f"Cannot begin a new round from phase '{self.phase}'. Expected 'round_complete'."
            )

        self.round_count += 1

        for player in self.players:
            player.reset_for_round()

        self.deck = Deck.full(self.config.suits)
        self.turn_count = 0
        self.pending_offer = None
        self.pending_draw_player = None
        self._undo_stack.clear()
        self._favors_updated_this_round = False

        # Create per-player event logs for this round
        for player_id in [0, 1]:
            self.info_state_history[player_id].append([])

        # Round 1: chance chooses starting player, then deals
        # Round 2+: alternate starting player deterministically, then deals
        if self.round_starting_player is None:
            self.phase = "choose_start"
            self.current_player = CHANCE_PLAYER
        else:
            self.round_starting_player = 1 - self.round_starting_player
            self._start_initial_deal_phase()

    def _start_initial_deal_phase(self) -> None:
        self.phase = "deal_initial"
        self.current_player = CHANCE_PLAYER
        self.pending_draw_player = None

    def _start_turn_draw_phase(self, player_id: int) -> None:
        if self._remaining_total() <= 0:
            self.phase = "play"
            self.current_player = player_id
            self.pending_draw_player = None
            return
        self.phase = "draw_for_turn"
        self.current_player = CHANCE_PLAYER
        self.pending_draw_player = player_id

    def _draw_specific_card(self, player_id: int, card: Card) -> bool:
        """Draw a specific card copy chosen during turn flow and record DrawCardEvent."""
        drawn = self.deck.draw_specific(card.suit)
        if drawn is None:
            return False

        self.players[player_id].hand.append(drawn)
        self._record_event(player_id, DrawCardEvent(turn=self.turn_count, card=drawn.suit))
        return True

    @staticmethod
    def _pop_matching_card(hand: List[Card], requested: Card) -> Optional[Tuple[int, Card]]:
        """Remove and return `(index, card)` for one matching hand card."""
        for idx, hand_card in enumerate(hand):
            if hand_card == requested:
                return idx, hand.pop(idx)
        return None

    @staticmethod
    def _restore_cards_to_deck(deck: Deck, cards: Sequence[Card]) -> None:
        """Add cards back into the deck multiset."""
        for card in cards:
            suit = card.suit
            deck.remaining_card_counts[suit] = deck.remaining_card_counts.get(suit, 0) + 1

    def _restore_pre_action_scalars(self, record: _UndoRecord) -> None:
        """Restore scalar/phase fields to their pre-action values."""
        self.phase = record.pre_phase
        self.current_player = record.pre_current_player
        self.turn_count = record.pre_turn_count
        self.pending_offer = record.pre_pending_offer
        self.pending_draw_player = record.pre_pending_draw_player
        self.round_starting_player = record.pre_round_starting_player

    def _undo_pop_events(self, player_id: int, count: int) -> None:
        if count <= 0:
            return
        round_events = self.info_state_history[player_id][-1]
        if len(round_events) < count:
            raise RuntimeError("Cannot undo events: event history shorter than expected.")
        del round_events[-count:]

    def _undo_restore_removed_cards(self, removed_cards: Sequence[Tuple[int, int, Card]]) -> None:
        """Reinsert removed cards in reverse removal order to preserve hand ordering."""
        for player_id, hand_index, card in reversed(removed_cards):
            self.players[player_id].hand.insert(hand_index, card)

    def _deal_initial_hands_from_sequence(self, dealt_cards: List[Card]) -> bool:
        """Apply initial deal sequence to both hands without emitting DrawCardEvent."""
        deal_count = self._initial_deal_cards_count()
        if len(dealt_cards) != deal_count:
            return False

        if not self.deck.remove_cards(dealt_cards):
            return False

        # Build initial hands directly from the deal order: even indices -> P0, odd -> P1.
        self.players[0].hand = [Card(card.suit) for idx, card in enumerate(dealt_cards) if idx % 2 == 0]
        self.players[1].hand = [Card(card.suit) for idx, card in enumerate(dealt_cards) if idx % 2 == 1]

        return True

    def _record_event(self, player_id: int, event: ObservationEvent) -> None:
        """Append an observation event to the player's current round log."""
        if not self.info_state_history[player_id]:
            raise RuntimeError("No round history exists for player; round may not be initialized.")
        self.info_state_history[player_id][-1].append(event)

    def _execute_chance_start(self, action: Action) -> bool:
        """Execute CHANCE_START: choose which player starts (round 1 only)."""
        if self.phase != "choose_start":
            return False
        if action.choice not in [0, 1]:
            return False
        self.round_starting_player = action.choice
        self._start_initial_deal_phase()
        return True

    def _execute_chance_deal(self, action: Action) -> bool:
        """Execute CHANCE_DEAL: deal initial cards to both players."""
        if self.phase != "deal_initial":
            return False
        if self.round_starting_player not in [0, 1]:
            return False
        if not self._deal_initial_hands_from_sequence(action.cards):
            return False

        # Emit ROUND_START observation for each player before any player action.
        # RoundInfoState.get_available_action_types() relies on this being event[0].
        for player_id in [0, 1]:
            self._record_event(
                player_id,
                RoundStartEvent(
                    turn=0,
                    initial_hand=tuple(c.suit for c in self.players[player_id].hand),
                    favors=_freeze_favors(self.favors),
                    starting_player=self.round_starting_player,
                    available_actions=self.config.available_actions,
                ),
            )

        # First turn begins with a chance draw for the starting player.
        self._start_turn_draw_phase(self.round_starting_player)
        return True

    def _execute_chance_draw(self, action: Action) -> bool:
        """Execute CHANCE_DRAW: draw one specific card for the pending player."""
        if self.phase != "draw_for_turn":
            return False
        if self.pending_draw_player not in [0, 1]:
            return False
        if action.choice != self.pending_draw_player:
            return False
        if len(action.cards) != 1:
            return False
        if not self._draw_specific_card(self.pending_draw_player, action.cards[0]):
            return False

        self.current_player = self.pending_draw_player
        self.pending_draw_player = None
        self.phase = "play"
        return True

    def _advance_after_action(self, action: Action):
        """Advance turn/player flow after a normal action (not an offer response)."""
        if action.action_type in OFFER_RESPONSES:
            # Offer: switch to opponent for their response (no turn increment, no draw)
            self.current_player = 1 - self.current_player
            self.phase = "play"
        else:
            # RESERVE, DISCARD or TRADE_OFF: increment turn, switch player, then chance draws.
            self.turn_count += 1
            self.current_player = 1 - self.current_player
            if not self.is_round_complete():
                self._start_turn_draw_phase(self.current_player)
            else:
                self.phase = "round_complete"
                self.pending_draw_player = None

    def _advance_after_offer_response(self):
        """Advance after an offer response: increment turn, keep responder, then draw."""
        self.turn_count += 1
        if not self.is_round_complete():
            self._start_turn_draw_phase(self.current_player)
        else:
            self.phase = "round_complete"
            self.pending_draw_player = None

    def execute_action(self, action: Action) -> bool:
        if self.current_player is None:
            raise RuntimeError("Cannot execute actions: round is complete. Begin the next round first.")

        pre_phase = self.phase
        pre_current_player = self._current_player
        pre_turn_count = self.turn_count
        pre_pending_offer = self.pending_offer
        pre_pending_draw_player = self.pending_draw_player
        pre_round_starting_player = self.round_starting_player

        def _make_undo_record(
            *,
            removed_cards: Sequence[Tuple[int, int, Card]] = (),
            added_used_action: Optional[Tuple[int, ActionType]] = None,
            events_appended_p0: int = 0,
            events_appended_p1: int = 0,
            dealt_cards: Sequence[Card] = (),
            pre_hand_p0: Sequence[Card] = (),
            pre_hand_p1: Sequence[Card] = (),
            draw_player: Optional[int] = None,
        ) -> _UndoRecord:
            return _UndoRecord(
                action_type=action.action_type,
                pre_phase=pre_phase,
                pre_current_player=pre_current_player,
                pre_turn_count=pre_turn_count,
                pre_pending_offer=pre_pending_offer,
                pre_pending_draw_player=pre_pending_draw_player,
                pre_round_starting_player=pre_round_starting_player,
                removed_cards=tuple(removed_cards),
                added_used_action=added_used_action,
                events_appended_p0=events_appended_p0,
                events_appended_p1=events_appended_p1,
                dealt_cards=tuple(dealt_cards),
                pre_hand_p0=tuple(pre_hand_p0),
                pre_hand_p1=tuple(pre_hand_p1),
                draw_player=draw_player,
            )

        # Route chance actions
        if action.action_type == ActionType.CHANCE_START:
            if not self._execute_chance_start(action):
                return False
            self._undo_stack.append(_make_undo_record())
            return True
        if action.action_type == ActionType.CHANCE_DEAL:
            pre_hand_p0 = tuple(self.players[0].hand)
            pre_hand_p1 = tuple(self.players[1].hand)
            if not self._execute_chance_deal(action):
                return False
            self._undo_stack.append(
                _make_undo_record(
                    events_appended_p0=1,
                    events_appended_p1=1,
                    dealt_cards=action.cards,
                    pre_hand_p0=pre_hand_p0,
                    pre_hand_p1=pre_hand_p1,
                )
            )
            return True
        if action.action_type == ActionType.CHANCE_DRAW:
            draw_player = self.pending_draw_player
            if not self._execute_chance_draw(action):
                return False
            if draw_player not in (0, 1):
                raise RuntimeError("Missing draw player while recording undo for CHANCE_DRAW.")
            self._undo_stack.append(
                _make_undo_record(
                    events_appended_p0=1 if draw_player == 0 else 0,
                    events_appended_p1=1 if draw_player == 1 else 0,
                    draw_player=draw_player,
                )
            )
            return True

        if self.current_player == CHANCE_PLAYER:
            raise RuntimeError("Cannot execute a player action on a chance node.")

        actor_id = self.current_player
        player = self.players[actor_id]
        opponent_id = 1 - actor_id

        pending_offer_type = self.pending_offer[1] if self.pending_offer is not None else None

        if not action.validate(player.hand, player.used_actions,
                               available_actions=self.config.available_actions,
                               pending_offer=pending_offer_type):
            return False

        if pending_offer_type is not None:
            result = self._execute_offer_response(action.choice)
            if result:
                self._advance_after_offer_response()
                self._undo_stack.append(_make_undo_record(events_appended_p0=1, events_appended_p1=1))
            return result

        removed_cards: List[Tuple[int, int, Card]] = []
        added_used_action: Optional[Tuple[int, ActionType]] = None
        events_appended_p0 = 0
        events_appended_p1 = 0

        if action.action_type == ActionType.RESERVE:
            card_match = self._pop_matching_card(player.hand, action.cards[0])
            if card_match is None:
                return False
            hand_idx, card = card_match
            removed_cards.append((actor_id, hand_idx, card))
            player.reserved_cards.append(card)
            player.used_actions.add(ActionType.RESERVE)
            added_used_action = (actor_id, ActionType.RESERVE)

            self._record_event(
                actor_id,
                OwnReserveEvent(turn=self.turn_count, card=card.suit),
            )
            self._record_event(
                opponent_id,
                OpponentReserveEvent(turn=self.turn_count),
            )
            events_appended_p0 = 1
            events_appended_p1 = 1

        elif action.action_type == ActionType.DISCARD:
            played_cards: List[Card] = []
            for requested in action.cards:
                card_match = self._pop_matching_card(player.hand, requested)
                if card_match is None:
                    self._undo_restore_removed_cards(removed_cards)
                    return False
                hand_idx, card = card_match
                removed_cards.append((actor_id, hand_idx, card))
                played_cards.append(card)

            revealed_card = played_cards[action.choice]
            hidden_card = played_cards[1 - action.choice]

            player.discarded_revealed.append(revealed_card)
            player.discarded_hidden.append(hidden_card)
            player.used_actions.add(ActionType.DISCARD)
            added_used_action = (actor_id, ActionType.DISCARD)

            self._record_event(
                actor_id,
                OwnDiscardEvent(
                    turn=self.turn_count,
                    revealed_card=revealed_card.suit,
                    hidden_card=hidden_card.suit,
                ),
            )
            self._record_event(
                opponent_id,
                OpponentDiscardEvent(turn=self.turn_count, revealed_card=revealed_card.suit),
            )
            events_appended_p0 = 1
            events_appended_p1 = 1

        elif action.action_type in (ActionType.TRADE_OFF, ActionType.SPLIT,
                                    ActionType.GIFT, ActionType.COMPETITION):
            played_cards: List[Card] = []
            for requested in action.cards:
                card_match = self._pop_matching_card(player.hand, requested)
                if card_match is None:
                    self._undo_restore_removed_cards(removed_cards)
                    return False
                hand_idx, card = card_match
                removed_cards.append((actor_id, hand_idx, card))
                played_cards.append(card)
            player.used_actions.add(action.action_type)
            added_used_action = (actor_id, action.action_type)
            suits = tuple(card.suit for card in played_cards)

            if action.action_type == ActionType.TRADE_OFF:
                # Both cards leave the round face down.
                player.discarded_hidden.extend(played_cards)
                own_event = OwnTradeOffEvent(turn=self.turn_count, cards=suits)
                opponent_event = OpponentTradeOffEvent(turn=self.turn_count)
            else:
                # An offer: the opponent responds before the turn ends.
                self.pending_offer = (actor_id, action.action_type, tuple(played_cards))
                if action.action_type == ActionType.SPLIT:
                    own_event = OwnSplitEvent(turn=self.turn_count, cards=suits)
                    opponent_event = OpponentSplitEvent(turn=self.turn_count, cards=suits)
                elif action.action_type == ActionType.GIFT:
                    own_event = OwnGiftEvent(turn=self.turn_count, cards=suits)
                    opponent_event = OpponentGiftEvent(turn=self.turn_count, cards=suits)
                else:
                    pairs = (suits[0:2], suits[2:4])
                    own_event = OwnCompetitionEvent(turn=self.turn_count, pairs=pairs)
                    opponent_event = OpponentCompetitionEvent(turn=self.turn_count, pairs=pairs)

            self._record_event(actor_id, own_event)
            self._record_event(opponent_id, opponent_event)
            events_appended_p0 = 1
            events_appended_p1 = 1
        else:
            return False

        self._advance_after_action(action)
        self._undo_stack.append(
            _make_undo_record(
                removed_cards=removed_cards,
                added_used_action=added_used_action,
                events_appended_p0=events_appended_p0,
                events_appended_p1=events_appended_p1,
            )
        )
        return True

    @staticmethod
    def _offer_split(offer_type: ActionType, cards: Tuple[Card, ...], choice: int
                     ) -> Tuple[Tuple[Card, ...], Tuple[Card, ...]]:
        """(cards the chooser gets, cards the offerer gets) for a response."""
        if offer_type == ActionType.COMPETITION:
            pairs = (cards[0:2], cards[2:4])
            return pairs[choice], pairs[1 - choice]
        chosen = (cards[choice],)
        rest = tuple(card for idx, card in enumerate(cards) if idx != choice)
        return chosen, rest

    def _execute_offer_response(self, choice: int) -> bool:
        if self.pending_offer is None:
            return False
        offerer_id, offer_type, cards = self.pending_offer
        _, n_choices = OFFER_RESPONSES[offer_type]
        if choice not in range(n_choices):
            return False

        chooser_id = self.current_player
        chooser_gets, offerer_gets = self._offer_split(offer_type, cards, choice)
        self.players[chooser_id].collected_cards.extend(chooser_gets)
        self.players[offerer_id].collected_cards.extend(offerer_gets)

        chooser_suits = tuple(card.suit for card in chooser_gets)
        offerer_suits = tuple(card.suit for card in offerer_gets)
        if offer_type == ActionType.SPLIT:
            choice_event = SplitChoiceEvent(
                turn=self.turn_count, chooser=chooser_id, choice_index=choice,
                chooser_gets=chooser_suits[0], offerer_gets=offerer_suits[0])
        elif offer_type == ActionType.GIFT:
            choice_event = GiftChoiceEvent(
                turn=self.turn_count, chooser=chooser_id, choice_index=choice,
                chooser_gets=chooser_suits[0], offerer_gets=offerer_suits)
        else:
            choice_event = CompetitionChoiceEvent(
                turn=self.turn_count, chooser=chooser_id, choice_index=choice,
                chooser_gets=chooser_suits, offerer_gets=offerer_suits)
        self._record_event(0, choice_event)
        self._record_event(1, choice_event)

        self.pending_offer = None
        return True

    def undo_move(self) -> None:
        """Undo the most recent `execute_action()` call.

        Undo is intentionally limited to action-level transitions. It is not
        valid once round-finalization side effects (`update_favors`) occurred.
        """
        if self._favors_updated_this_round:
            raise RuntimeError(
                "Cannot undo actions after update_favors() for this round. "
                "Use preview_round_resolution()/terminal_payoff() for search."
            )
        if not self._undo_stack:
            raise RuntimeError("No moves to undo.")

        record = self._undo_stack.pop()
        action_type = record.action_type

        if action_type == ActionType.CHANCE_START:
            self._restore_pre_action_scalars(record)
            return

        if action_type == ActionType.CHANCE_DEAL:
            self._undo_pop_events(0, record.events_appended_p0)
            self._undo_pop_events(1, record.events_appended_p1)
            self.players[0].hand = list(record.pre_hand_p0)
            self.players[1].hand = list(record.pre_hand_p1)
            self._restore_cards_to_deck(self.deck, record.dealt_cards)
            self._restore_pre_action_scalars(record)
            return

        if action_type == ActionType.CHANCE_DRAW:
            if record.draw_player not in (0, 1):
                raise RuntimeError("Malformed undo record for CHANCE_DRAW.")
            self._undo_pop_events(record.draw_player, 1)
            hand = self.players[record.draw_player].hand
            if not hand:
                raise RuntimeError("Cannot undo CHANCE_DRAW: expected drawn card in hand.")
            card = hand.pop()
            self._restore_cards_to_deck(self.deck, (card,))
            self._restore_pre_action_scalars(record)
            return

        if action_type in (ActionType.SPLIT_RESPONSE, ActionType.GIFT_RESPONSE,
                           ActionType.COMPETITION_RESPONSE):
            if record.pre_pending_offer is None:
                raise RuntimeError(f"Malformed undo record for {action_type.name}.")
            offerer_id, offer_type, cards = record.pre_pending_offer
            chooser_id = record.pre_current_player
            # The response's choice is not stored; the counts don't depend on it.
            chooser_n, offerer_n = (len(part) for part in self._offer_split(offer_type, cards, 0))
            for pid, n, who in ((chooser_id, chooser_n, "chooser"), (offerer_id, offerer_n, "offerer")):
                collected = self.players[pid].collected_cards
                if len(collected) < n:
                    raise RuntimeError(f"Cannot undo {action_type.name}: {who} has too few collected cards.")
                del collected[-n:]
            self._undo_pop_events(0, record.events_appended_p0)
            self._undo_pop_events(1, record.events_appended_p1)
            self._restore_pre_action_scalars(record)
            return

        if action_type == ActionType.RESERVE:
            actor_id = record.pre_current_player
            actor = self.players[actor_id]
            if not actor.reserved_cards:
                raise RuntimeError("Cannot undo RESERVE: reserved pile is empty.")
            actor.reserved_cards.pop()
            if record.added_used_action is not None:
                actor.used_actions.remove(record.added_used_action[1])
            self._undo_restore_removed_cards(record.removed_cards)
            self._undo_pop_events(0, record.events_appended_p0)
            self._undo_pop_events(1, record.events_appended_p1)
            self._restore_pre_action_scalars(record)
            return

        if action_type == ActionType.DISCARD:
            actor_id = record.pre_current_player
            actor = self.players[actor_id]
            if not actor.discarded_revealed or not actor.discarded_hidden:
                raise RuntimeError("Cannot undo DISCARD: discarded piles are empty.")
            actor.discarded_revealed.pop()
            actor.discarded_hidden.pop()
            if record.added_used_action is not None:
                actor.used_actions.remove(record.added_used_action[1])
            self._undo_restore_removed_cards(record.removed_cards)
            self._undo_pop_events(0, record.events_appended_p0)
            self._undo_pop_events(1, record.events_appended_p1)
            self._restore_pre_action_scalars(record)
            return

        if action_type == ActionType.TRADE_OFF:
            actor_id = record.pre_current_player
            actor = self.players[actor_id]
            if len(actor.discarded_hidden) < 2:
                raise RuntimeError("Cannot undo TRADE_OFF: hidden discard pile has fewer than 2 cards.")
            del actor.discarded_hidden[-2:]
            if record.added_used_action is not None:
                actor.used_actions.remove(record.added_used_action[1])
            self._undo_restore_removed_cards(record.removed_cards)
            self._undo_pop_events(0, record.events_appended_p0)
            self._undo_pop_events(1, record.events_appended_p1)
            self._restore_pre_action_scalars(record)
            return

        if action_type in (ActionType.SPLIT, ActionType.GIFT, ActionType.COMPETITION):
            actor_id = record.pre_current_player
            actor = self.players[actor_id]
            if record.added_used_action is not None:
                actor.used_actions.remove(record.added_used_action[1])
            self._undo_restore_removed_cards(record.removed_cards)
            self._undo_pop_events(0, record.events_appended_p0)
            self._undo_pop_events(1, record.events_appended_p1)
            self._restore_pre_action_scalars(record)
            return

        raise RuntimeError(f"Unhandled undo action type: {action_type!r}")

    def update_favors(self):
        # Fail-fast: favors only make sense after the round has completed.
        if self.phase != "round_complete" or not self.is_round_complete():
            raise RuntimeError(
                f"update_favors() can only be called at end-of-round "
                f"(phase='round_complete' and round complete). Got phase='{self.phase}', "
                f"turn_count={self.turn_count}."
            )

        preview = self.preview_round_resolution()

        for player in self.players:
            player.collected_cards.extend(player.reserved_cards)
            player.reserved_cards = []

        self.favors = dict(preview.final_favors)

        round_end_event = RoundEndEvent(
            turn=self.turn_count,
            p0_reserved_revealed=preview.p0_reserved_revealed,
            p1_reserved_revealed=preview.p1_reserved_revealed,
            final_favors=preview.final_favors,
        )
        self._record_event(0, round_end_event)
        self._record_event(1, round_end_event)
        self._favors_updated_this_round = True

    def preview_round_resolution(self) -> RoundResolutionPreview:
        """Compute end-of-round favors/winner without mutating state.

        This allows evaluators (e.g. tree traversal payoff code) to compute
        terminal utility without calling `update_favors()`.
        """
        if not self.is_round_complete():
            raise RuntimeError(
                "preview_round_resolution() requires a completed round "
                f"(turn_count={self.turn_count}, turns_per_round={self.config.turns_per_round})."
            )

        p0_reserved = tuple(card.suit for card in self.players[0].reserved_cards)
        p1_reserved = tuple(card.suit for card in self.players[1].reserved_cards)

        final_favors = dict(self.favors)
        count_p0 = self.players[0].get_suit_count(include_reserved=True)
        count_p1 = self.players[1].get_suit_count(include_reserved=True)
        for suit in self.config.suits:
            if count_p0[suit] > count_p1[suit]:
                final_favors[suit] = -1
            elif count_p1[suit] > count_p0[suit]:
                final_favors[suit] = 1

        winner = self._winner_from_favors(final_favors)
        return RoundResolutionPreview(
            p0_reserved_revealed=p0_reserved,
            p1_reserved_revealed=p1_reserved,
            final_favors=_freeze_favors(final_favors),
            winner=winner,
        )

    def terminal_payoff(self, player_id: int = 0) -> float:
        """Return terminal utility (+1/-1/0) for a completed round.

        Utility is computed from the pure preview path (no mutation).
        """
        if player_id not in (0, 1):
            raise ValueError(f"player_id must be 0 or 1 (got {player_id!r}).")
        preview = self.preview_round_resolution()
        if preview.winner is None:
            return 0.0
        return 1.0 if preview.winner == player_id else -1.0

    def _winner_from_favors(self, favors: Dict[object, int]) -> Optional[int]:
        """Compute winner from a favors map."""
        favor_points = [0, 0]
        favor_counts = [0, 0]

        for suit, favor in favors.items():
            if favor == -1:
                favor_counts[0] += 1
                favor_points[0] += suit.rank
            elif favor == 1:
                favor_counts[1] += 1
                favor_points[1] += suit.rank

        # Check favor points condition (if enabled)
        if self.config.win_favor_points is not None:
            if favor_points[0] >= self.config.win_favor_points:
                return 0
            if favor_points[1] >= self.config.win_favor_points:
                return 1

        # Check geisha count condition
        if favor_counts[0] >= self.config.win_geisha_count:
            return 0
        if favor_counts[1] >= self.config.win_geisha_count:
            return 1

        return None

    def check_winner(self) -> Optional[int]:
        return self._winner_from_favors(self.favors)

    def is_round_complete(self) -> bool:
        return self.turn_count >= self.config.turns_per_round

    # -------------------------------------------------------------------
    # Legal action enumeration (generators for memory efficiency)
    # -------------------------------------------------------------------

    def get_legal_actions(self) -> Generator[Action, None, None]:
        """Yield all legal actions from the current state.

        For chance nodes, yields CHANCE_START/CHANCE_DEAL/CHANCE_DRAW actions.
        For player nodes, yields deduplicated player actions.
        """
        if self.current_player is None:
            raise RuntimeError("No legal actions: round is complete. Begin the next round first.")
        if self.current_player == CHANCE_PLAYER:
            if self.phase == "choose_start":
                yield Action(ActionType.CHANCE_START, [], choice=0)
                yield Action(ActionType.CHANCE_START, [], choice=1)
            elif self.phase == "deal_initial":
                deal_count = self._initial_deal_cards_count()
                yield from self.deck.iter_chance_deal_actions(deal_count)
            elif self.phase == "draw_for_turn":
                if self.pending_draw_player not in (0, 1):
                    raise RuntimeError("Missing pending draw player for CHANCE_DRAW phase.")
                yield from self.deck.iter_chance_draw_actions(self.pending_draw_player)
        elif self.pending_offer is not None:
            yield from iter_offer_response_actions(self.pending_offer[1])
        else:
            yield from self._get_player_legal_actions()

    def _get_player_legal_actions(self) -> Generator[Action, None, None]:
        """Yield deduplicated legal actions for the current player.

        Same-suit cards are interchangeable, so we yield one representative
        action per unique suit combination.
        """
        if self.current_player in (None, CHANCE_PLAYER):
            raise RuntimeError("Cannot enumerate player legal actions when no player is to act.")
        player = self.players[self.current_player]
        yield from iter_player_legal_actions(
            hand=player.hand,
            used_actions=player.used_actions,
            available_actions=self.config.available_actions,
        )

    def sample_chance_action(self, rng: Optional[random.Random] = None) -> Action:
        """Sample a random chance action (for simulation, avoids full enumeration)."""
        rand = rng if rng is not None else random
        if self.phase == "choose_start":
            return Action(ActionType.CHANCE_START, [], choice=rand.randint(0, 1))
        elif self.phase == "deal_initial":
            need = self._initial_deal_cards_count()
            return self.deck.sample_chance_deal_action(need, rand)
        elif self.phase == "draw_for_turn":
            if self.pending_draw_player not in [0, 1]:
                raise ValueError("Missing pending draw player for CHANCE_DRAW phase.")
            return self.deck.sample_chance_draw_action(self.pending_draw_player, rand)
        else:
            raise ValueError(f"Cannot sample chance action in phase '{self.phase}'")

    def _count_distinct_initial_deals(self) -> int:
        """Count distinct initial deals (pairs of hands)."""
        deal_count = self._initial_deal_cards_count()
        return self.deck.count_distinct_deals(deal_count)

    def get_chance_actions_with_probs(self) -> Generator[Tuple[Action, float], None, None]:
        """Yield (action, probability) pairs for chance nodes.

        For CHANCE_START: yields 2 actions with 0.5 probability each.
        For CHANCE_DEAL: yields distinct initial deals with exact probabilities.
        For CHANCE_DRAW: yields distinct draw suits with probabilities by remaining count.

        Raises ValueError if called on a non-chance node.
        """
        cp = self.current_player
        if cp is None:
            raise RuntimeError("No chance actions: round is complete. Begin the next round first.")
        if cp != CHANCE_PLAYER:
            raise ValueError(f"Not a chance node (current_player={cp})")

        if self.phase == "choose_start":
            yield (Action(ActionType.CHANCE_START, [], choice=0), 0.5)
            yield (Action(ActionType.CHANCE_START, [], choice=1), 0.5)
        elif self.phase == "deal_initial":
            deal_count = self._initial_deal_cards_count()
            yield from self.deck.iter_chance_deal_actions_with_probs(deal_count)
        elif self.phase == "draw_for_turn":
            if self.pending_draw_player not in [0, 1]:
                raise ValueError("Missing pending draw player for CHANCE_DRAW phase.")
            yield from self.deck.iter_chance_draw_actions_with_probs(self.pending_draw_player)
        else:
            raise ValueError(f"Unexpected phase for chance node: {self.phase}")

    def get_info_state(self, player_id: int) -> RoundInfoState:
        if not isinstance(player_id, int) or player_id not in (0, 1):
            raise ValueError(f"player_id must be 0 or 1 (got {player_id!r}).")
        if not self.info_state_history[player_id]:
            return RoundInfoState(player_id=player_id, round_number=self.round_count, events=())
        # Return detached event objects so in-process agent mutation cannot alter history.
        events = tuple(replace(event) for event in self.info_state_history[player_id][-1])
        round_number = len(self.info_state_history[player_id])
        return RoundInfoState(player_id=player_id, round_number=round_number, events=events)

    def get_debug_snapshot(self, player_id: int) -> dict:
        """Return a dict snapshot intended for debugging/printing (not game logic)."""
        return snapshot_game_state(self, player_id)

    def to_key(self) -> "GameStateKey":
        """Return a hashable, value-based snapshot for use as a dict/set key."""

        def _cards(lst):
            return tuple(sorted((c.suit for c in lst), key=lambda s: s.value))

        po = None
        if self.pending_offer is not None:
            offerer_id, offer_type, cards = self.pending_offer
            po = (offerer_id, offer_type, tuple(c.suit for c in cards))

        p0, p1 = self.players[0], self.players[1]
        return GameStateKey(
            phase=self.phase,
            current_player=self._current_player,
            turn_count=self.turn_count,
            round_count=self.round_count,
            round_starting_player=self.round_starting_player,
            pending_draw_player=self.pending_draw_player,
            winner=self.winner,
            deck=frozenset(self.deck.nonzero_suit_counts().items()),
            favors=frozenset(self.favors.items()),
            pending_offer=po,
            p0_hand=_cards(p0.hand),
            p0_reserved=_cards(p0.reserved_cards),
            p0_collected=_cards(p0.collected_cards),
            p0_discarded_revealed=_cards(p0.discarded_revealed),
            p0_discarded_hidden=_cards(p0.discarded_hidden),
            p0_used_actions=frozenset(p0.used_actions),
            p1_hand=_cards(p1.hand),
            p1_reserved=_cards(p1.reserved_cards),
            p1_collected=_cards(p1.collected_cards),
            p1_discarded_revealed=_cards(p1.discarded_revealed),
            p1_discarded_hidden=_cards(p1.discarded_hidden),
            p1_used_actions=frozenset(p1.used_actions),
        )

    def __str__(self) -> str:
        return format_game_state_one_line(self)

    def pretty_print(self, box_width: int = 58) -> str:
        return pretty_print_game_state(self, box_width)
