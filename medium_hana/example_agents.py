"""
Example agent implementations for the Hanamikoji game.
Students can use these as starting points or references.
"""
from __future__ import annotations

from medium_hana import Agent, Action, ActionType, Card, Suit, RoundInfoState
from typing import List, Dict


class GreedyAgent(Agent):
    """
    A simple greedy agent that tries to collect high-value geishas.
    Strategy: Reserve high-value cards, discard low-value cards.
    """

    def get_action_distribution(self, info_state: RoundInfoState):
        return {self._select_action(info_state): 1.0}

    def _select_action(self, info_state: RoundInfoState) -> Action:
        # Check if we need to respond to a split
        pending_split = info_state.get_pending_split()
        if pending_split is not None:
            # Always pick the higher rank card
            choice = 0 if pending_split[0].suit.rank >= pending_split[1].suit.rank else 1
            return Action(ActionType.SPLIT_RESPONSE, [], choice=choice)

        hand, used_actions = info_state.get_current_hand_and_actions()

        # Sort hand by rank (highest first)
        sorted_hand = sorted(hand, key=lambda c: c.suit.rank, reverse=True)

        # Reserve highest value card if possible
        if ActionType.RESERVE not in used_actions:
            return Action(ActionType.RESERVE, [sorted_hand[0]])

        # Discard lowest value cards if possible
        if ActionType.DISCARD not in used_actions and len(hand) >= 2:
            lowest_cards = sorted_hand[-2:]
            reveal_choice = 1 if lowest_cards[1].suit.rank <= lowest_cards[0].suit.rank else 0
            return Action(ActionType.DISCARD, lowest_cards, choice=reveal_choice)

        # Split with remaining cards
        if ActionType.SPLIT not in used_actions and len(hand) >= 2:
            return Action(ActionType.SPLIT, hand[:2])

        # Fallback: all actions used, shouldn't happen in normal gameplay
        raise ValueError("GreedyAgent: No valid action available")


class BalancedAgent(Agent):
    """
    An agent that tries to spread its influence across multiple geishas
    rather than focusing on just high-value ones.
    """

    def get_action_distribution(self, info_state: RoundInfoState):
        return {self._select_action(info_state): 1.0}

    def _select_action(self, info_state: RoundInfoState) -> Action:
        # Check if we need to respond to a split
        pending_split = info_state.get_pending_split()
        if pending_split is not None:
            return self._select_split_response(info_state, pending_split)

        hand, used_actions = info_state.get_current_hand_and_actions()
        my_collected, opponent_collected = info_state.get_collected_cards()

        # Count current collection for each suit
        my_suits = {suit: 0 for suit in Suit}
        opp_suits = {suit: 0 for suit in Suit}

        for card in my_collected:
            my_suits[card.suit] += 1
        for card in opponent_collected:
            opp_suits[card.suit] += 1

        # Reserve action: pick a card where we need to catch up or maintain lead
        if ActionType.RESERVE not in used_actions:
            contested_cards = [
                c for c in hand
                if my_suits[c.suit] <= opp_suits[c.suit]
            ]

            if contested_cards:
                best_card = max(contested_cards, key=lambda c: c.suit.rank)
                return Action(ActionType.RESERVE, [best_card])
            else:
                best_card = max(hand, key=lambda c: c.suit.rank)
                return Action(ActionType.RESERVE, [best_card])

        # Discard action: discard suits where opponent already has strong lead
        if ActionType.DISCARD not in used_actions and len(hand) >= 2:
            discard_candidates = []
            for card in hand:
                if opp_suits[card.suit] - my_suits[card.suit] >= 2:
                    discard_candidates.append(card)

            if len(discard_candidates) >= 2:
                cards_to_discard = sorted(discard_candidates, key=lambda c: c.suit.rank)[:2]
                reveal_choice = 0 if cards_to_discard[0].suit.rank <= cards_to_discard[1].suit.rank else 1
                return Action(ActionType.DISCARD, cards_to_discard, choice=reveal_choice)
            else:
                sorted_hand = sorted(hand, key=lambda c: c.suit.rank)
                return Action(ActionType.DISCARD, sorted_hand[:2], choice=0)

        # Split action: offer a difficult choice to opponent
        if ActionType.SPLIT not in used_actions and len(hand) >= 2:
            sorted_hand = sorted(hand, key=lambda c: c.suit.rank, reverse=True)

            best_pair = None
            min_diff = float('inf')

            for i in range(len(sorted_hand)):
                for j in range(i + 1, len(sorted_hand)):
                    diff = abs(sorted_hand[i].suit.rank - sorted_hand[j].suit.rank)
                    if diff < min_diff:
                        min_diff = diff
                        best_pair = [sorted_hand[i], sorted_hand[j]]

            return Action(ActionType.SPLIT, best_pair)

        # Fallback: all actions used, shouldn't happen in normal gameplay
        raise ValueError("BalancedAgent: No valid action available")

    def _select_split_response(self, info_state: RoundInfoState, cards: List[Card]) -> Action:
        """Choose which card to take from a split."""
        my_collected, _ = info_state.get_collected_cards()

        my_suits = {suit: 0 for suit in Suit}
        for card in my_collected:
            my_suits[card.suit] += 1

        scores = []
        for card in cards:
            score = card.suit.rank
            score += my_suits[card.suit] * 2
            scores.append(score)

        choice = 0 if scores[0] >= scores[1] else 1
        return Action(ActionType.SPLIT_RESPONSE, [], choice=choice)


class CardCountingAgent(Agent):
    """
    An agent that estimates unseen cards from the current visible information.
    This is more advanced and demonstrates information management.
    """

    def _get_visible_suit_counts(self, info_state: RoundInfoState) -> Dict[Suit, int]:
        """Count every currently visible card copy by suit from this perspective."""
        hand, _ = info_state.get_current_hand_and_actions()
        my_collected, opponent_collected = info_state.get_collected_cards()
        my_reserved = info_state.get_my_reserved()
        my_revealed, my_hidden, opponent_revealed = info_state.get_discarded_cards()
        pending_split = info_state.get_pending_split() or []

        visible_cards = (
            hand
            + my_collected
            + opponent_collected
            + my_reserved
            + my_revealed
            + my_hidden
            + opponent_revealed
            + pending_split
        )

        visible_counts = {suit: 0 for suit in Suit}
        for card in visible_cards:
            visible_counts[card.suit] += 1
        return visible_counts

    def get_unseen_suits(self, info_state: RoundInfoState) -> Dict[Suit, int]:
        """Calculate unseen card copies per suit from current visible information."""
        total_counts = {suit: suit.rank for suit in Suit}
        visible_counts = self._get_visible_suit_counts(info_state)

        unseen = {}
        for suit in Suit:
            unseen[suit] = max(0, total_counts[suit] - visible_counts[suit])

        return unseen

    def get_action_distribution(self, info_state: RoundInfoState):
        return {self._select_action(info_state): 1.0}

    def _select_action(self, info_state: RoundInfoState) -> Action:
        # Check if we need to respond to a split
        pending_split = info_state.get_pending_split()
        if pending_split is not None:
            return self._select_split_response(info_state, pending_split)

        hand, used_actions = info_state.get_current_hand_and_actions()
        my_collected, opponent_collected = info_state.get_collected_cards()

        my_suits = {suit: 0 for suit in Suit}
        opp_suits = {suit: 0 for suit in Suit}

        for card in my_collected:
            my_suits[card.suit] += 1
        for card in opponent_collected:
            opp_suits[card.suit] += 1

        unseen = self.get_unseen_suits(info_state)

        # Reserve: Pick card where we can potentially win the suit
        if ActionType.RESERVE not in used_actions:
            winnable_cards = []

            for card in hand:
                suit = card.suit
                max_we_can_get = my_suits[suit] + 1 + unseen[suit]

                if max_we_can_get > opp_suits[suit]:
                    winnable_cards.append((card, card.suit.rank))

            if winnable_cards:
                best_card = max(winnable_cards, key=lambda x: x[1])[0]
                return Action(ActionType.RESERVE, [best_card])
            else:
                best_card = max(hand, key=lambda c: c.suit.rank)
                return Action(ActionType.RESERVE, [best_card])

        # Discard: Remove cards from suits we're likely to lose anyway
        if ActionType.DISCARD not in used_actions and len(hand) >= 2:
            likely_losses = []

            for card in hand:
                suit = card.suit
                if opp_suits[suit] > my_suits[suit] + unseen[suit]:
                    likely_losses.append(card)

            if len(likely_losses) >= 2:
                cards_to_discard = sorted(likely_losses, key=lambda c: c.suit.rank)[:2]
                reveal_choice = 0 if cards_to_discard[0].suit.rank <= cards_to_discard[1].suit.rank else 1
                return Action(ActionType.DISCARD, cards_to_discard, choice=reveal_choice)
            else:
                sorted_hand = sorted(hand, key=lambda c: c.suit.rank)
                return Action(ActionType.DISCARD, sorted_hand[:2], choice=0)

        # Split: Try to give opponent a difficult choice
        if ActionType.SPLIT not in used_actions and len(hand) >= 2:
            return Action(ActionType.SPLIT, hand[:2])

        # Fallback: all actions used, shouldn't happen in normal gameplay
        raise ValueError("CardCountingAgent: No valid action available")

    def _select_split_response(self, info_state: RoundInfoState, cards: List[Card]) -> Action:
        """Choose which card to take from a split."""
        my_collected, opp_collected = info_state.get_collected_cards()

        my_suits = {suit: 0 for suit in Suit}
        opp_suits = {suit: 0 for suit in Suit}

        for card in my_collected:
            my_suits[card.suit] += 1
        for card in opp_collected:
            opp_suits[card.suit] += 1

        scores = []
        for card in cards:
            suit = card.suit
            score = card.suit.rank * 10

            if my_suits[suit] + 1 > opp_suits[suit]:
                score += 20
            if my_suits[suit] + 1 == opp_suits[suit]:
                score += 10

            scores.append(score)

        choice = 0 if scores[0] >= scores[1] else 1
        return Action(ActionType.SPLIT_RESPONSE, [], choice=choice)


if __name__ == "__main__":
    from medium_hana import Game, RandomAgent

    print("Testing example agents...")
    print("="*60)

    agents_list = [
        ("Random", RandomAgent()),
        ("Greedy", GreedyAgent()),
        ("Balanced", BalancedAgent()),
        ("CardCounting", CardCountingAgent()),
    ]

    results = {name: 0 for name, _ in agents_list}

    games_per_matchup = 20

    for i, (name1, agent1) in enumerate(agents_list):
        for name2, agent2 in agents_list[i+1:]:
            wins1 = 0
            wins2 = 0

            for _ in range(games_per_matchup):
                game = Game(agent1, agent2, verbose=False)
                winner = game.play_game()
                if winner == 0:
                    wins1 += 1
                elif winner == 1:
                    wins2 += 1

            results[name1] += wins1
            results[name2] += wins2

            print(f"{name1:15s} vs {name2:15s}: {wins1:2d}-{wins2:2d}")

    print("\n" + "="*60)
    print("Overall Results:")
    print("="*60)
    for name, wins in sorted(results.items(), key=lambda x: -x[1]):
        print(f"{name:15s}: {wins:3d} wins")
