"""Debug rendering helpers for the engine.

These functions intentionally live outside `engine.state.GameState` to keep the
state machine focused on rules and transitions.
"""

from __future__ import annotations

from typing import List, Sequence

from .enums import ActionType
from .models import Card


def snapshot_game_state(state, player_id: int) -> dict:
    """Return a detached, debug-friendly snapshot of the current game state.

    This is for visualization/logging only. It returns fresh `Card` objects so
    callers cannot mutate live engine state by editing the snapshot.
    """
    if not isinstance(player_id, int) or player_id not in (0, 1):
        raise ValueError(f"player_id must be 0 or 1 (got {player_id!r}).")
    opponent_id = 1 - player_id

    def clone_cards(cards: Sequence[Card]) -> List[Card]:
        return [Card(card.suit) for card in cards]

    pending_split_cards = clone_cards(state.pending_split[1]) if state.pending_split else None
    pending_offer = None
    if state.pending_offer is not None:
        offerer_id, offer_type, cards = state.pending_offer
        pending_offer = (offerer_id, offer_type, clone_cards(cards))

    return {
        "player_id": player_id,
        "round": state.round_count,
        "turn": state.turn_count,
        "hand": clone_cards(state.players[player_id].hand),
        "used_actions": state.players[player_id].used_actions.copy(),
        "my_collected": clone_cards(state.players[player_id].collected_cards),
        "my_reserved": clone_cards(state.players[player_id].reserved_cards),
        "my_reserved_count": len(state.players[player_id].reserved_cards),
        "my_discarded_revealed": clone_cards(state.players[player_id].discarded_revealed),
        "my_discarded_hidden": clone_cards(state.players[player_id].discarded_hidden),
        "opponent_collected": clone_cards(state.players[opponent_id].collected_cards),
        "opponent_reserved_count": len(state.players[opponent_id].reserved_cards),
        "opponent_discarded_revealed": clone_cards(state.players[opponent_id].discarded_revealed),
        "opponent_used_actions": state.players[opponent_id].used_actions.copy(),
        "opponent_hand_size": len(state.players[opponent_id].hand),
        "favors": state.favors.copy(),
        "pending_split": pending_split_cards,
        "pending_offer": pending_offer,
        "deck_size": state.deck.total(),
    }


def format_game_state_one_line(state) -> str:
    """Format a compact one-line summary of game state."""
    if state.phase in ("choose_start", "deal_initial", "draw_for_turn"):
        return f"GameState(phase={state.phase}, round={state.round_count})"

    p0_hand = len(state.players[0].hand)
    p1_hand = len(state.players[1].hand)
    pending = f"{state.pending_offer[1].value}_pending" if state.pending_offer else ""
    cp = state.current_player
    cp_str = "-" if cp is None else str(cp)

    return (
        f"GameState(R{state.round_count} T{state.turn_count}/{state.config.turns_per_round} "
        f"P{cp_str} hands=[{p0_hand},{p1_hand}] remaining={state.deck.total()}"
        f"{' ' + pending if pending else ''})"
    )


def pretty_print_game_state(state, box_width: int = 58) -> str:
    """Detailed visual display of game state.

    Returns a string with box-drawing characters showing:
    - Round/turn/phase header
    - Each player's hand, collected, reserved, discarded cards
    - Geisha favors
    - Deck info and pending split state
    """

    def pad_line(text: str) -> str:
        return text + " " * (box_width - len(text))

    def format_cards(cards: List[Card], indent: str = "     ") -> List[str]:
        if not cards:
            return []
        return [f"{indent}• {c.suit.name} (rank {c.suit.rank})" for c in cards]

    lines = []

    # Header
    lines.append("╔" + "═" * box_width + "╗")
    if state.phase in ("choose_start", "deal_initial", "draw_for_turn"):
        header = f" PHASE: {state.phase.upper()}  •  ROUND {state.round_count}"
    elif state.phase == "round_complete":
        header = (
            f" ROUND {state.round_count}  •  TURN {state.turn_count}/{state.config.turns_per_round}  •  ROUND COMPLETE"
        )
    else:
        header = (
            f" ROUND {state.round_count}  •  TURN {state.turn_count}/{state.config.turns_per_round}  •  "
            f"P{state.current_player}'s turn"
        )
    lines.append("║" + pad_line(header) + "║")
    lines.append("╠" + "═" * box_width + "╣")

    # Player sections
    for pid in (0, 1):
        player = state.players[pid]
        lines.append("║" + pad_line(f" PLAYER {pid}:") + "║")

        # Hand
        lines.append("║" + pad_line(f"   Hand ({len(player.hand)} cards):") + "║")
        for card_line in format_cards(player.hand):
            lines.append("║" + pad_line(card_line) + "║")

        # Collected
        lines.append("║" + pad_line(f"   Collected ({len(player.collected_cards)} cards):") + "║")
        for card_line in format_cards(player.collected_cards):
            lines.append("║" + pad_line(card_line) + "║")

        # Reserved
        lines.append("║" + pad_line(f"   Reserved ({len(player.reserved_cards)} cards):") + "║")
        for card_line in format_cards(player.reserved_cards):
            lines.append("║" + pad_line(card_line) + "║")

        # Discarded (only show if variant supports DISCARD or TRADE_OFF)
        if state.config.available_actions & {ActionType.DISCARD, ActionType.TRADE_OFF}:
            if player.discarded_revealed:
                lines.append("║" + pad_line("   Discarded (revealed):") + "║")
                for card_line in format_cards(player.discarded_revealed):
                    lines.append("║" + pad_line(card_line) + "║")
            if player.discarded_hidden:
                lines.append("║" + pad_line("   Discarded (hidden):") + "║")
                for card_line in format_cards(player.discarded_hidden):
                    lines.append("║" + pad_line(card_line) + "║")

        # Actions used
        used = ", ".join(a.name for a in player.used_actions) or "None"
        lines.append("║" + pad_line(f"   Actions used: {used}") + "║")

        lines.append("╠" + "═" * box_width + "╣")

    # Geisha favors
    lines.append("║" + pad_line(" GEISHA FAVORS:") + "║")
    for suit in state.config.suits:
        favor = state.favors[suit]
        status = "P0" if favor == -1 else "P1" if favor == 1 else "---"
        lines.append("║" + pad_line(f"   {suit.name:12s} (rank {suit.rank}): [{status}]") + "║")

    lines.append("╠" + "═" * box_width + "╣")

    # Remaining cards and pending split
    lines.append("║" + pad_line(f" Remaining cards: {state.deck.total()}") + "║")
    if state.pending_offer:
        offerer, offer_type, cards = state.pending_offer
        card_names = ", ".join(c.suit.name for c in cards)
        if offer_type == ActionType.COMPETITION:
            card_names = (", ".join(c.suit.name for c in cards[0:2]) + " | "
                          + ", ".join(c.suit.name for c in cards[2:4]))
        lines.append("║" + pad_line(f" Pending {offer_type.value} from P{offerer}: [{card_names}]") + "║")

    lines.append("╚" + "═" * box_width + "╝")
    return "\n".join(lines)

