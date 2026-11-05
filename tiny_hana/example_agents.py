"""
Example AI agents for Tiny Hanamikoji.

These agents demonstrate different strategies for the simplified game.
"""
from __future__ import annotations

from tiny_hana import Agent, Action, ActionType, Card, Suit, RoundInfoState
from typing import List


class GreedyAgent(Agent):
    """
    Greedy strategy: Always tries to collect high-value geisha (GEISHA_3).

    Strategy:
    - Reserves highest-rank cards
    - In SPLIT, offers lower-value cards when possible
    - When responding to split, picks higher rank
    """

    def get_action_distribution(self, info_state: RoundInfoState):
        return {self._select_action(info_state): 1.0}

    def _select_action(self, info_state: RoundInfoState) -> Action:
        # Check if we need to respond to a split
        pending_split = info_state.get_pending_split()
        if pending_split is not None:
            # Always pick higher rank card
            choice = 0 if pending_split[0].suit.rank >= pending_split[1].suit.rank else 1
            return Action(ActionType.SPLIT_RESPONSE, [], choice=choice)

        hand, used_actions = info_state.get_current_hand_and_actions()

        # Sort hand by rank (highest first)
        sorted_hand = sorted(hand, key=lambda c: c.suit.rank, reverse=True)

        # Reserve highest value card if possible
        if ActionType.RESERVE not in used_actions:
            return Action(ActionType.RESERVE, [sorted_hand[0]])

        # Split with remaining cards
        if ActionType.SPLIT not in used_actions and len(hand) >= 2:
            return Action(ActionType.SPLIT, hand[:2])

        # Fallback: both actions used, shouldn't happen in normal gameplay
        raise ValueError("GreedyAgent: No valid action available")


class BalancedAgent(Agent):
    """
    Balanced strategy: Tries to spread influence across multiple geishas.

    Strategy:
    - Avoids focusing too much on one geisha
    - Reserves cards for geishas where we need advantage
    """

    def get_action_distribution(self, info_state: RoundInfoState):
        return {self._select_action(info_state): 1.0}

    def _select_action(self, info_state: RoundInfoState) -> Action:
        # Check if we need to respond to a split
        pending_split = info_state.get_pending_split()
        if pending_split is not None:
            return self._select_split_response(info_state, pending_split)

        hand, used_actions = info_state.get_current_hand_and_actions()
        my_collected, _ = info_state.get_collected_cards()
        my_reserved = info_state.get_my_reserved()

        # Count what we already have
        my_counts = {suit: 0 for suit in Suit}
        for card in my_collected + my_reserved:
            my_counts[card.suit] += 1

        # Find card of suit we have least of
        hand_by_need = sorted(hand, key=lambda c: my_counts[c.suit])

        # Reserve card we need most
        if ActionType.RESERVE not in used_actions:
            return Action(ActionType.RESERVE, [hand_by_need[0]])

        # Split with remaining cards
        if ActionType.SPLIT not in used_actions and len(hand) >= 2:
            return Action(ActionType.SPLIT, hand[:2])

        # Fallback: both actions used, shouldn't happen in normal gameplay
        raise ValueError("BalancedAgent: No valid action available")

    def _select_split_response(self, info_state: RoundInfoState, cards: List[Card]) -> Action:
        """Choose which card to take from a split."""
        my_collected, _ = info_state.get_collected_cards()
        my_reserved = info_state.get_my_reserved()

        my_counts = {suit: 0 for suit in Suit}
        for card in my_collected + my_reserved:
            my_counts[card.suit] += 1

        # Pick card of suit we have less of
        choice = 0 if my_counts[cards[0].suit] <= my_counts[cards[1].suit] else 1
        return Action(ActionType.SPLIT_RESPONSE, [], choice=choice)


class CardCountingAgent(Agent):
    """
    Card counting strategy: Tracks which cards have been revealed.

    Strategy:
    - Knows exactly which cards are still hidden
    - Makes decisions based on probability of opponent having certain cards
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

        # Count visible cards per suit
        visible_cards = hand + my_collected + opponent_collected
        visible_counts = {suit: 0 for suit in Suit}
        for card in visible_cards:
            visible_counts[card.suit] += 1

        # Score each card based on scarcity (fewer visible = more valuable)
        hand_scored = sorted(hand, key=lambda c: (
            visible_counts[c.suit],   # Fewer visible = higher priority
            c.suit.rank               # Higher rank as tiebreaker
        ))

        # Reserve most valuable card
        if ActionType.RESERVE not in used_actions:
            return Action(ActionType.RESERVE, [hand_scored[0]])

        # Split with remaining cards
        if ActionType.SPLIT not in used_actions and len(hand) >= 2:
            return Action(ActionType.SPLIT, hand[:2])

        # Fallback: both actions used, shouldn't happen in normal gameplay
        raise ValueError("CardCountingAgent: No valid action available")

    def _select_split_response(self, info_state: RoundInfoState, cards: List[Card]) -> Action:
        """Choose which card to take from a split."""
        my_collected, opponent_collected = info_state.get_collected_cards()
        my_reserved = info_state.get_my_reserved()

        my_counts = {suit: 0 for suit in Suit}
        opp_counts = {suit: 0 for suit in Suit}

        for card in my_collected + my_reserved:
            my_counts[card.suit] += 1
        for card in opponent_collected:
            opp_counts[card.suit] += 1

        # Pick card where we're most behind or tied
        card0_need = opp_counts[cards[0].suit] - my_counts[cards[0].suit]
        card1_need = opp_counts[cards[1].suit] - my_counts[cards[1].suit]

        choice = 0 if card0_need >= card1_need else 1
        return Action(ActionType.SPLIT_RESPONSE, [], choice=choice)


class AdaptiveAgent(Agent):
    """
    Adaptive strategy: Changes tactics based on current favor state.

    Strategy:
    - If winning, plays defensively
    - If losing, plays aggressively
    """

    def get_action_distribution(self, info_state: RoundInfoState):
        return {self._select_action(info_state): 1.0}

    def _select_action(self, info_state: RoundInfoState) -> Action:
        # Check if we need to respond to a split
        pending_split = info_state.get_pending_split()
        if pending_split is not None:
            return self._select_split_response(info_state, pending_split)

        hand, used_actions = info_state.get_current_hand_and_actions()
        favors = info_state.get_favors()

        # Count favors
        my_favors = sum(1 for f in favors.values() if f == -1)
        opp_favors = sum(1 for f in favors.values() if f == 1)

        # If we're winning (have 1 favor already), play defensively
        if my_favors >= 1:
            # Reserve high-value cards
            sorted_hand = sorted(hand, key=lambda c: c.suit.rank, reverse=True)
            if ActionType.RESERVE not in used_actions:
                return Action(ActionType.RESERVE, [sorted_hand[0]])

        # If we're losing, play aggressively
        elif opp_favors >= 1:
            # Try to target geishas we can still win
            priority_cards = [c for c in hand if favors[c.suit] != 1]
            if priority_cards and ActionType.RESERVE not in used_actions:
                best = max(priority_cards, key=lambda c: c.suit.rank)
                return Action(ActionType.RESERVE, [best])

        # Default: balanced approach
        if ActionType.RESERVE not in used_actions:
            return Action(ActionType.RESERVE, [hand[0]])

        # Split with remaining cards
        if ActionType.SPLIT not in used_actions and len(hand) >= 2:
            return Action(ActionType.SPLIT, hand[:2])

        # Fallback: both actions used, shouldn't happen in normal gameplay
        raise ValueError("AdaptiveAgent: No valid action available")

    def _select_split_response(self, info_state: RoundInfoState, cards: List[Card]) -> Action:
        """Choose which card to take from a split."""
        favors = info_state.get_favors()

        # Pick card for geisha we don't control
        if favors[cards[0].suit] != -1 and favors[cards[1].suit] == -1:
            choice = 0
        elif favors[cards[1].suit] != -1 and favors[cards[0].suit] == -1:
            choice = 1
        else:
            # Otherwise pick higher rank
            choice = 0 if cards[0].suit.rank >= cards[1].suit.rank else 1

        return Action(ActionType.SPLIT_RESPONSE, [], choice=choice)


if __name__ == "__main__":
    # Test the agents against each other
    from tiny_hana import Game, RandomAgent

    print("Testing example agents...")
    print("="*60)

    agents_list = [
        ("Random", RandomAgent()),
        ("Greedy", GreedyAgent()),
        ("Balanced", BalancedAgent()),
        ("CardCounting", CardCountingAgent()),
        ("Adaptive", AdaptiveAgent()),
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
