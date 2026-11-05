"""
Interactive Text User Interface for Hanamikoji

Allows a human player to play against AI agents.
Supports multiple game variants (Full Hanamikoji, Hanamikoji, Tiny Hanamikoji).
"""
from __future__ import annotations

from engine import Agent, Action, ActionType, Card, RoundInfoState, RandomAgent, Game, GameConfig, CHANCE_PLAYER
from typing import List, Optional
import random
import sys


def select_variant() -> dict:
    """Let user choose which game variant to play."""
    print("\n" + "="*60)
    print("  SELECT GAME VARIANT")
    print("="*60)
    print("\n  [1] Hanamikoji       - 5 geishas, 3 actions, 11 cards")
    print("  [2] Tiny Hanamikoji  - 3 geishas, 2 actions, 7 cards")
    print("  [3] Full Hanamikoji  - 7 geishas, 4 actions, 21 cards (the published game)")

    while True:
        try:
            choice = int(input("\nSelect variant [1-3]: "))
            if choice == 1:
                from medium_hana.medium_hana import Suit, GAME_CONFIG, GameState
                from medium_hana.example_agents import GreedyAgent, BalancedAgent, CardCountingAgent
                return {
                    'Suit': Suit,
                    'GAME_CONFIG': GAME_CONFIG,
                    'GameState': GameState,
                    'agents': [
                        ("RandomAgent", RandomAgent()),
                        ("GreedyAgent", GreedyAgent()),
                        ("BalancedAgent", BalancedAgent()),
                        ("CardCountingAgent", CardCountingAgent()),
                    ],
                }
            elif choice == 2:
                from tiny_hana.tiny_hana import Suit, GAME_CONFIG, GameState
                from tiny_hana.example_agents import GreedyAgent, BalancedAgent, CardCountingAgent, AdaptiveAgent
                return {
                    'Suit': Suit,
                    'GAME_CONFIG': GAME_CONFIG,
                    'GameState': GameState,
                    'agents': [
                        ("RandomAgent", RandomAgent()),
                        ("GreedyAgent", GreedyAgent()),
                        ("BalancedAgent", BalancedAgent()),
                        ("CardCountingAgent", CardCountingAgent()),
                        ("AdaptiveAgent", AdaptiveAgent()),
                    ],
                }
            elif choice == 3:
                from full_hana.full_hana import Suit, GAME_CONFIG, GameState
                from full_hana.example_agents import GreedyAgent, CardCountingAgent
                return {
                    'Suit': Suit,
                    'GAME_CONFIG': GAME_CONFIG,
                    'GameState': GameState,
                    'agents': [
                        ("RandomAgent", RandomAgent()),
                        ("GreedyAgent", GreedyAgent()),
                        ("CardCountingAgent", CardCountingAgent()),
                    ],
                }
            else:
                print("Please enter 1, 2 or 3")
        except ValueError:
            print("Please enter a valid number")
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting...")
            sys.exit(0)


class HumanAgent(Agent):
    """Interactive agent controlled by human player via text interface."""

    def __init__(self, name="Human", game_config: GameConfig = None):
        self.name = name
        self.game_config = game_config
        self.box_width = 58  # Total width inside the box (without borders)

    def pad_line(self, text: str) -> str:
        """Pad a line to fit exactly in the box."""
        # Calculate how many spaces needed to reach box_width
        spaces_needed = self.box_width - len(text)
        if spaces_needed < 0:
            # Text too long, truncate it
            return text[:self.box_width]
        return text + " " * spaces_needed

    def display_card(self, card: Card, index: int = None) -> str:
        """Format a card for display."""
        prefix = f"[{index}] " if index is not None else ""
        return f"{prefix}{card.suit.name} (rank {card.suit.rank})"

    def display_hand(self, hand: List[Card]) -> None:
        """Display the player's hand."""
        print("\n" + "="*60)
        print("YOUR HAND:")
        print("="*60)
        for i, card in enumerate(hand):
            print(f"  {self.display_card(card, i)}")
        print()

    def display_game_state(self, info_state: RoundInfoState) -> None:
        """Display the current game state."""
        # Extract state from info_state
        hand, used_actions = info_state.get_current_hand_and_actions()
        my_collected, opponent_collected = info_state.get_collected_cards()
        my_reserved = info_state.get_my_reserved()
        opponent_reserved_count = info_state.get_opponent_reserved_count()
        my_revealed, my_hidden, opponent_revealed = info_state.get_discarded_cards()
        opponent_used_actions = info_state.get_opponent_used_actions()
        favors = info_state.get_favors()
        turn = info_state.get_current_turn()

        turns_per_round = self.game_config.turns_per_round if self.game_config else 6

        print("\n" + "╔"+"═"*self.box_width+"╗")
        print("║" + self.pad_line(f" ROUND {info_state.round_number}  •  TURN {turn+1}/{turns_per_round}") + "║")
        print("╠"+"═"*self.box_width+"╣")

        # Your info
        print("║" + self.pad_line(" YOUR STATUS:") + "║")
        print("║" + self.pad_line(f"   Collected: {len(my_collected)} cards") + "║")
        for card in my_collected:
            print("║" + self.pad_line(f"     • {card.suit.name} (rank {card.suit.rank})") + "║")

        print("║" + self.pad_line(f"   Reserved: {len(my_reserved)} card(s)") + "║")
        for card in my_reserved:
            print("║" + self.pad_line(f"     • {card.suit.name} (rank {card.suit.rank})") + "║")

        if my_revealed:
            print("║" + self.pad_line("   Discarded & Revealed:") + "║")
            for card in my_revealed:
                print("║" + self.pad_line(f"     • {card.suit.name} (rank {card.suit.rank})") + "║")

        if my_hidden:
            print("║" + self.pad_line("   Discarded & Hidden:") + "║")
            for card in my_hidden:
                print("║" + self.pad_line(f"     • {card.suit.name} (rank {card.suit.rank})") + "║")

        print("╠"+"═"*self.box_width+"╣")

        # Opponent info
        print("║" + self.pad_line(" OPPONENT STATUS:") + "║")
        print("║" + self.pad_line(f"   Collected: {len(opponent_collected)} cards") + "║")
        for card in opponent_collected:
            print("║" + self.pad_line(f"     • {card.suit.name} (rank {card.suit.rank})") + "║")

        print("║" + self.pad_line(f"   Reserved: {opponent_reserved_count} card(s) (hidden)") + "║")

        if opponent_revealed:
            print("║" + self.pad_line("   Discarded & Revealed:") + "║")
            for card in opponent_revealed:
                print("║" + self.pad_line(f"     • {card.suit.name} (rank {card.suit.rank})") + "║")

        print("╠"+"═"*self.box_width+"╣")

        # Favors
        print("║" + self.pad_line(" GEISHA FAVORS:") + "║")
        for suit in favors:
            favor = favors[suit]
            status = "YOU" if favor == -1 else "OPP" if favor == 1 else "---"
            print("║" + self.pad_line(f"   {suit.name:12s} (rank {suit.rank}): [{status}]") + "║")

        print("╠"+"═"*self.box_width+"╣")

        # Actions used
        print("║" + self.pad_line(" ACTIONS USED:") + "║")
        you_actions = ', '.join(a.name for a in used_actions) or 'None'
        print("║" + self.pad_line(f"   You:      {you_actions}") + "║")
        opp_actions = ', '.join(a.name for a in opponent_used_actions) or 'None'
        print("║" + self.pad_line(f"   Opponent: {opp_actions}") + "║")

        print("╚"+"═"*self.box_width+"╝")

    def get_int_input(self, prompt: str, min_val: int, max_val: int) -> int:
        """Get validated integer input from user."""
        while True:
            try:
                choice = input(prompt)
                value = int(choice)
                if min_val <= value <= max_val:
                    return value
                else:
                    print(f"  Please enter a number between {min_val} and {max_val}")
            except ValueError:
                print(f"  Please enter a valid number")
            except (KeyboardInterrupt, EOFError):
                print("\n\nExiting game...")
                sys.exit(0)

    def get_action_distribution(self, info_state: RoundInfoState):
        return {self._select_action(info_state): 1.0}

    def _select_action(self, info_state: RoundInfoState) -> Action:
        """Get a legal action from the human player (asks again if not legal)."""
        legal = set(info_state.get_legal_actions())
        while True:
            action = self._ask_action(info_state)
            if action in legal:
                return action
            print("  That move is not legal here (e.g. not enough cards left for it); try again.")

    def _ask_action(self, info_state: RoundInfoState) -> Action:
        # Check if we need to respond to an offer (split, gift or competition)
        pending_offer = info_state.get_pending_offer()
        if pending_offer is not None:
            offer_type, cards = pending_offer
            if offer_type == ActionType.SPLIT:
                return self._select_split_response(cards)
            return self._select_offer_response(offer_type, cards)

        self.display_game_state(info_state)

        hand, used_actions = info_state.get_current_hand_and_actions()
        self.display_hand(hand)

        # Get available normal actions (responses are not chosen from this menu)
        responses = (ActionType.SPLIT_RESPONSE, ActionType.GIFT_RESPONSE, ActionType.COMPETITION_RESPONSE)
        variant_actions = sorted(info_state.get_available_action_types(), key=lambda a: a.name)
        available_actions = [
            a for a in variant_actions
            if a not in used_actions and a not in responses
        ]

        if not available_actions:
            raise ValueError("No available actions for human player.")

        # Show available actions
        print("AVAILABLE ACTIONS:")
        action_menu = []
        for i, action_type in enumerate(available_actions, 1):
            if action_type == ActionType.RESERVE:
                desc = "Reserve 1 card (hidden, you get it)"
            elif action_type == ActionType.DISCARD:
                desc = "Discard 2 cards (reveal 1, hide 1)"
            elif action_type == ActionType.SPLIT:
                desc = "Split 2 cards (opponent picks 1, you get the other)"
            elif action_type == ActionType.TRADE_OFF:
                desc = "Trade-off: discard 2 cards face down"
            elif action_type == ActionType.GIFT:
                desc = "Gift: offer 3 cards (opponent takes 1, you keep 2)"
            elif action_type == ActionType.COMPETITION:
                desc = "Competition: offer 2 pairs (opponent takes 1 pair, you get the other)"
            else:
                desc = action_type.name
            action_menu.append((action_type, desc))
            print(f"  [{i}] {action_type.name}: {desc}")

        # Get action choice
        choice = self.get_int_input("\nChoose action [1-{}]: ".format(len(action_menu)),
                                     1, len(action_menu))
        chosen_action_type = action_menu[choice - 1][0]

        # Execute chosen action
        if chosen_action_type == ActionType.RESERVE:
            print("\n-> RESERVE: Choose a card to reserve (hidden)")
            card_idx = self.get_int_input(f"  Card to reserve [0-{len(hand)-1}]: ",
                                          0, len(hand) - 1)
            return Action(ActionType.RESERVE, [hand[card_idx]])

        elif chosen_action_type == ActionType.DISCARD:
            if len(hand) < 2:
                print("  Not enough cards to discard!")
                return self._select_action(info_state)

            print("\n-> DISCARD: Choose 2 cards to discard")
            print("  (You'll then choose which one to reveal to opponent)")

            card1_idx = self.get_int_input(f"  First card [0-{len(hand)-1}]: ",
                                           0, len(hand) - 1)

            # Show remaining cards
            print("\n  Remaining cards:")
            for i, card in enumerate(hand):
                if i != card1_idx:
                    print(f"    {self.display_card(card, i)}")

            card2_idx = self.get_int_input(f"  Second card [0-{len(hand)-1}]: ",
                                           0, len(hand) - 1)

            while card2_idx == card1_idx:
                print("  Must choose a different card!")
                card2_idx = self.get_int_input(f"  Second card [0-{len(hand)-1}]: ",
                                               0, len(hand) - 1)

            cards = [hand[card1_idx], hand[card2_idx]]

            print("\n  Cards to discard:")
            print(f"    [0] {cards[0].suit.name} (rank {cards[0].suit.rank})")
            print(f"    [1] {cards[1].suit.name} (rank {cards[1].suit.rank})")

            reveal_choice = self.get_int_input("  Which card to REVEAL to opponent? [0-1]: ",
                                               0, 1)

            print(f"  -> Revealing: {cards[reveal_choice].suit.name}")
            print(f"  -> Hiding: {cards[1-reveal_choice].suit.name}")

            return Action(ActionType.DISCARD, cards, choice=reveal_choice)

        elif chosen_action_type == ActionType.SPLIT:
            if len(hand) < 2:
                print("  Not enough cards to split!")
                return self._select_action(info_state)

            print("\n-> SPLIT: Choose 2 cards (opponent will pick 1, you get the other)")

            card1_idx = self.get_int_input(f"  First card [0-{len(hand)-1}]: ",
                                           0, len(hand) - 1)

            # Show remaining cards
            print("\n  Remaining cards:")
            for i, card in enumerate(hand):
                if i != card1_idx:
                    print(f"    {self.display_card(card, i)}")

            card2_idx = self.get_int_input(f"  Second card [0-{len(hand)-1}]: ",
                                           0, len(hand) - 1)

            while card2_idx == card1_idx:
                print("  Must choose a different card!")
                card2_idx = self.get_int_input(f"  Second card [0-{len(hand)-1}]: ",
                                               0, len(hand) - 1)

            cards = [hand[card1_idx], hand[card2_idx]]

            print("\n  Cards to offer:")
            print(f"    {cards[0].suit.name} (rank {cards[0].suit.rank})")
            print(f"    {cards[1].suit.name} (rank {cards[1].suit.rank})")

            return Action(ActionType.SPLIT, cards)

        elif chosen_action_type == ActionType.TRADE_OFF:
            print("\n-> TRADE-OFF: Choose 2 cards to discard face down")
            return Action(ActionType.TRADE_OFF, self._pick_cards(hand, 2))

        elif chosen_action_type == ActionType.GIFT:
            print("\n-> GIFT: Choose 3 cards (opponent takes 1, you keep the other 2)")
            return Action(ActionType.GIFT, self._pick_cards(hand, 3))

        elif chosen_action_type == ActionType.COMPETITION:
            print("\n-> COMPETITION: Choose 4 cards; the first two form pair A, the last two pair B")
            print("   (opponent takes one pair, you get the other)")
            return Action(ActionType.COMPETITION, self._pick_cards(hand, 4))

    def _pick_cards(self, hand: List[Card], n: int) -> List[Card]:
        """Ask for `n` different cards from the hand, by index."""
        picked: List[int] = []
        while len(picked) < n:
            idx = self.get_int_input(f"  Card {len(picked) + 1} of {n} [0-{len(hand)-1}]: ", 0, len(hand) - 1)
            if idx in picked:
                print("  Must choose a different card!")
                continue
            picked.append(idx)
        return [hand[i] for i in picked]

    def _select_offer_response(self, offer_type, cards: List[Card]) -> Action:
        """Get a GIFT / COMPETITION response from the human player."""
        name = "GIFT" if offer_type == ActionType.GIFT else "COMPETITION"
        print("\n" + "!"*60)
        print(f"OPPONENT PLAYED {name} - YOU MUST CHOOSE!")
        print("!"*60)
        if offer_type == ActionType.GIFT:
            print("\nTake one card; the opponent keeps the other two:")
            for i, card in enumerate(cards):
                print(f"  [{i}] {card.suit.name} (rank {card.suit.rank})")
            choice = self.get_int_input("\nWhich card do you want? [0-2]: ", 0, 2)
            print(f"\n-> You chose: {cards[choice].suit.name}")
            return Action(ActionType.GIFT_RESPONSE, [], choice=choice)
        pairs = [cards[0:2], cards[2:4]]
        print("\nTake one pair; the opponent gets the other:")
        for i, pair in enumerate(pairs):
            print(f"  [{i}] {pair[0].suit.name} + {pair[1].suit.name}")
        choice = self.get_int_input("\nWhich pair do you want? [0-1]: ", 0, 1)
        print(f"\n-> You chose: {pairs[choice][0].suit.name} + {pairs[choice][1].suit.name}")
        return Action(ActionType.COMPETITION_RESPONSE, [], choice=choice)

    def _select_split_response(self, cards: List[Card]) -> Action:
        """Get split response from human player."""
        print("\n" + "!"*60)
        print("OPPONENT PLAYED SPLIT - YOU MUST CHOOSE!")
        print("!"*60)
        print("\nOpponent offers you one of these cards:")
        print(f"  [0] {cards[0].suit.name} (rank {cards[0].suit.rank})")
        print(f"  [1] {cards[1].suit.name} (rank {cards[1].suit.rank})")

        choice = self.get_int_input("\nWhich card do you want? [0-1]: ", 0, 1)

        print(f"\n-> You chose: {cards[choice].suit.name}")
        print(f"-> Opponent gets: {cards[1-choice].suit.name}")

        return Action(ActionType.SPLIT_RESPONSE, [], choice=choice)


def select_opponent(variant_name: str, agent_choices: list) -> Agent:
    """Let user choose which AI opponent to play against."""
    print("\n" + "="*60)
    print(f"  {variant_name.upper()} - Interactive Game")
    print("="*60)
    print("\nChoose your opponent:")
    for i, (name, _agent) in enumerate(agent_choices, 1):
        print(f"  [{i}] {name}")

    num_choices = len(agent_choices)

    while True:
        try:
            choice = int(input(f"\nSelect opponent [1-{num_choices}]: "))
            if 1 <= choice <= num_choices:
                name, agent = agent_choices[choice - 1]
                print(f"\n You will play against: {name}")
                return agent
            else:
                print(f"Please enter a number between 1 and {num_choices}")
        except ValueError:
            print("Please enter a valid number")
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting...")
            sys.exit(0)


def _count_cards_by_suit(cards: List[Card]) -> dict:
    """Count card copies by suit."""
    counts = {}
    for card in cards:
        counts[card.suit] = counts.get(card.suit, 0) + 1
    return counts


def _format_collected_reserved(collected: int, reserved: int) -> str:
    """Format suit count as either 'X' or 'X + Y (reserved)'."""
    if reserved == 0:
        return str(collected)
    return f"{collected} + {reserved} (reserved)"


def _build_favor_board_rows(state, reserved_snapshot: dict = None) -> dict:
    """Build per-suit display rows plus aggregate favor/point totals."""
    p0_total = _count_cards_by_suit(state.players[0].collected_cards)
    p1_total = _count_cards_by_suit(state.players[1].collected_cards)

    if reserved_snapshot:
        p0_reserved = _count_cards_by_suit(reserved_snapshot['player_0'])
        p1_reserved = _count_cards_by_suit(reserved_snapshot['player_1'])
    else:
        p0_reserved = {}
        p1_reserved = {}

    rows = []
    your_geishas = 0
    opp_geishas = 0
    neutral_geishas = 0
    your_points = 0
    opp_points = 0

    for suit, favor in state.favors.items():
        if favor == -1:
            status = "YOU"
            your_geishas += 1
            your_points += suit.rank
        elif favor == 1:
            status = "OPP"
            opp_geishas += 1
            opp_points += suit.rank
        else:
            status = "---"
            neutral_geishas += 1

        your_reserved_count = p0_reserved.get(suit, 0)
        opp_reserved_count = p1_reserved.get(suit, 0)
        your_total = p0_total.get(suit, 0)
        opp_total = p1_total.get(suit, 0)
        your_collected = your_total - your_reserved_count
        opp_collected = opp_total - opp_reserved_count

        your_count_str = _format_collected_reserved(your_collected, your_reserved_count)
        opp_count_str = _format_collected_reserved(opp_collected, opp_reserved_count)

        rows.append({
            'suit': suit,
            'status': status,
            'count_info': f"You:{your_count_str} Opp:{opp_count_str}",
        })

    return {
        'rows': rows,
        'your_geishas': your_geishas,
        'opp_geishas': opp_geishas,
        'neutral_geishas': neutral_geishas,
        'your_points': your_points,
        'opp_points': opp_points,
    }


def display_final_results(state, winner: int, reserved_snapshot: dict = None):
    """Display the final game results."""
    box_width = 85

    def pad_line(text: str) -> str:
        spaces_needed = box_width - len(text)
        if spaces_needed < 0:
            return text[:box_width]
        return text + " " * spaces_needed

    print("\n" + "╔"+"═"*box_width+"╗")
    print("║" + pad_line(" "*21 + "GAME OVER!") + "║")
    print("╠"+"═"*box_width+"╣")

    if winner == 0:
        print("║" + pad_line(" "*18 + "YOU WIN!") + "║")
    elif winner == 1:
        print("║" + pad_line(" "*15 + "OPPONENT WINS") + "║")
    else:
        print("║" + pad_line(" "*22 + "DRAW!") + "║")

    print("╠"+"═"*box_width+"╣")

    # Show final favor state
    print("║" + pad_line(" FINAL GEISHA FAVORS:") + "║")

    board = _build_favor_board_rows(state, reserved_snapshot)
    for row in board['rows']:
        suit = row['suit']
        status = row['status']
        count_info = row['count_info']
        print("║" + pad_line(f"   {suit.name:12s} (rank {suit.rank}): [{status}] ({count_info})") + "║")

    print("╠"+"═"*box_width+"╣")
    print("║" + pad_line(f" YOU:      {board['your_geishas']} geishas, {board['your_points']} points") + "║")
    print("║" + pad_line(f" OPPONENT: {board['opp_geishas']} geishas, {board['opp_points']} points") + "║")
    print("╚"+"═"*box_width+"╝")


def display_opponent_action(action: Action, player_id: int):
    """Display what the opponent just played (respecting privacy)."""
    print("\n" + "-"*60)
    print(f"OPPONENT'S TURN (Player {player_id}):")

    if action.action_type == ActionType.RESERVE:
        print("  -> Played RESERVE (card is hidden)")

    elif action.action_type == ActionType.DISCARD:
        revealed_card = action.cards[action.choice]
        print(f"  -> Played DISCARD")
        print(f"     Revealed: {revealed_card.suit.name} (rank {revealed_card.suit.rank})")
        print(f"     Hidden: (card hidden)")

    elif action.action_type == ActionType.SPLIT:
        print(f"  -> Played SPLIT, offering you:")
        print(f"     {action.cards[0].suit.name} (rank {action.cards[0].suit.rank})")
        print(f"     {action.cards[1].suit.name} (rank {action.cards[1].suit.rank})")
        print("  (You will choose on your next turn)")

    elif action.action_type == ActionType.TRADE_OFF:
        print("  -> Played TRADE-OFF (2 cards discarded face down)")

    elif action.action_type == ActionType.GIFT:
        print("  -> Played GIFT, offering you one of:")
        for card in action.cards:
            print(f"     {card.suit.name} (rank {card.suit.rank})")
        print("  (You will choose on your next turn)")

    elif action.action_type == ActionType.COMPETITION:
        print("  -> Played COMPETITION, offering you one of these pairs:")
        for pair in (action.cards[0:2], action.cards[2:4]):
            print(f"     {pair[0].suit.name} + {pair[1].suit.name}")
        print("  (You will choose on your next turn)")

    print("-"*60)


def display_round_end(state, round_num: int, game_config: GameConfig, reserved_snapshot: dict = None):
    """Display round-end status when there's no winner yet."""
    box_width = 85

    def pad_line(text: str) -> str:
        spaces_needed = box_width - len(text)
        if spaces_needed < 0:
            return text[:box_width]
        return text + " " * spaces_needed

    print("\n" + "╔"+"═"*box_width+"╗")
    print("║" + pad_line(f" END OF ROUND {round_num} - NO WINNER YET") + "║")
    print("╠"+"═"*box_width+"╣")

    print("║" + pad_line(" GEISHA FAVORS:") + "║")

    board = _build_favor_board_rows(state, reserved_snapshot)
    for row in board['rows']:
        suit = row['suit']
        status = row['status']
        count_info = row['count_info']
        print("║" + pad_line(f"   {suit.name:12s} (rank {suit.rank}): [{status}] ({count_info})") + "║")

    print("╠"+"═"*box_width+"╣")
    print("║" + pad_line(f" YOU:      {board['your_geishas']} geishas, {board['your_points']} points") + "║")
    print("║" + pad_line(f" OPPONENT: {board['opp_geishas']} geishas, {board['opp_points']} points") + "║")
    print("║" + pad_line(f" NEUTRAL:  {board['neutral_geishas']} geishas") + "║")
    print("╠"+"═"*box_width+"╣")

    win_favor = game_config.win_favor_points
    win_geisha = game_config.win_geisha_count

    if win_favor is not None and board['your_points'] >= win_favor:
        print("║" + pad_line(f" YOU WIN! ({win_favor}+ points)") + "║")
    elif win_favor is not None and board['opp_points'] >= win_favor:
        print("║" + pad_line(f" OPPONENT WINS! ({win_favor}+ points)") + "║")
    elif board['your_geishas'] >= win_geisha:
        print("║" + pad_line(f" YOU WIN! ({win_geisha}+ geishas)") + "║")
    elif board['opp_geishas'] >= win_geisha:
        print("║" + pad_line(f" OPPONENT WINS! ({win_geisha}+ geishas)") + "║")
    else:
        need_you = max(win_geisha - board['your_geishas'], 0)
        need_opp = max(win_geisha - board['opp_geishas'], 0)
        print("║" + pad_line(f" TIED - Next round begins!") + "║")
        if win_favor is not None:
            print("║" + pad_line(f" You need {need_you} more geisha(s) or {win_favor-board['your_points']} more points") + "║")
            print("║" + pad_line(f" Opp needs {need_opp} more geisha(s) or {win_favor-board['opp_points']} more points") + "║")
        else:
            print("║" + pad_line(f" You need {need_you} more geisha(s)") + "║")
            print("║" + pad_line(f" Opp needs {need_opp} more geisha(s)") + "║")

    print("╚"+"═"*box_width+"╝")

    if state.winner is None:
        print("\nPress Enter to continue to next round...")
        try:
            input()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGame interrupted. Goodbye!")
            sys.exit(0)


def play_interactive_game(human: HumanAgent, opponent: Agent,
                          GameState, game_config: GameConfig,
                          seed: Optional[int] = None) -> tuple:
    """Custom game loop for interactive mode with better feedback.

    Uses the unified action model where SPLIT_RESPONSE is a separate decision.
    """
    state = GameState()
    agents = [human, opponent]
    max_rounds = 10
    reserved_snapshot = None
    rng = random.Random(seed)
    rounds_played = 0

    while state.winner is None and rounds_played < max_rounds:
        # Start the next round if the previous one completed.
        if state.phase == "round_complete":
            state._begin_round()

        # Resolve chance nodes (choose starting player, deal cards)
        while state.current_player == CHANCE_PLAYER:
            state.execute_action(state.sample_chance_action(rng))

        # Play decisions until round is complete
        while not state.is_round_complete():
            if state.current_player == CHANCE_PLAYER:
                state.execute_action(state.sample_chance_action(rng))
                continue

            current_player = state.current_player
            if current_player is None:
                raise RuntimeError("Round is complete; interactive loop should have exited already.")
            agent = agents[current_player]
            is_split_response = state.pending_offer is not None   # responding to any offer

            # Show context for what's happening
            if current_player == 1:  # Opponent's turn/response
                if not is_split_response:
                    print("\n" + "="*60)
                    print("OPPONENT'S TURN BEGINS")
                    print("="*60)
                    human_info = state.get_info_state(0)
                    human.display_game_state(human_info)

            # Get and execute action
            info_state = state.get_info_state(current_player)
            distribution = agent.get_action_distribution(info_state)
            action = Game.sample_action_from_distribution(distribution, rng)
            if action is None:
                raise ValueError(f"Agent {current_player} returned None action.")

            if not state.execute_action(action):
                raise ValueError(f"Invalid action by player {current_player}")

            # Show opponent's action to human (after execution)
            if current_player == 1:
                if action.action_type not in (ActionType.SPLIT_RESPONSE, ActionType.GIFT_RESPONSE,
                                              ActionType.COMPETITION_RESPONSE):
                    display_opponent_action(action, current_player)

        # Round complete - capture reserved cards before they're cleared
        reserved_snapshot = {
            'player_0': state.players[0].reserved_cards.copy(),
            'player_1': state.players[1].reserved_cards.copy()
        }

        # Update favors and check winner
        state.update_favors()
        state.winner = state.check_winner()
        rounds_played += 1

        # Display round end status
        if state.winner is None:
            display_round_end(state, state.round_count, game_config, reserved_snapshot)

    winner = state.winner if state.winner is not None else -1
    return winner, state, reserved_snapshot


def main():
    """Run the interactive game."""
    try:
        # Select variant
        variant = select_variant()
        game_config = variant['GAME_CONFIG']

        # Select opponent
        opponent = select_opponent(game_config.name, variant['agents'])

        # Build reminder text dynamically
        num_actions = len(game_config.available_actions)
        turns = game_config.turns_per_round

        print("\n" + "="*60)
        print("Game starting! You are Player 0.")
        print("="*60)
        print("\nREMINDER:")
        print(f"  Each round has {turns} turns ({num_actions} per player)")
        print("  You draw 1 card at the start of each turn")
        print("  Each action can only be used once per round")
        win_msg = f"  Win by controlling {game_config.win_geisha_count} geishas"
        if game_config.win_favor_points is not None:
            win_msg += f" OR {game_config.win_favor_points}+ favor points"
        print(win_msg)
        print("\nPress Enter to begin...")
        input()

        # Create agents and play
        human = HumanAgent("You", game_config=game_config)
        winner, final_state, reserved_snapshot = play_interactive_game(
            human, opponent, variant['GameState'], game_config)

        # Show results
        display_final_results(final_state, winner, reserved_snapshot)

        print("\nThanks for playing!")

    except (KeyboardInterrupt, EOFError):
        print("\n\nGame interrupted. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
