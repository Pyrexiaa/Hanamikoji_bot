# Tiny Hanamikoji - Simplified Version

A smaller, faster variant of Hanamikoji perfect for quick games and testing AI strategies.

## Game Differences from the Hanamikoji Variant (medium_hana)

The comparisons below are with medium_hana. The published game, with 7
geishas and four actions, is full_hana.

### Deck Size
- **Medium version**: 11 cards (5 geishas, ranks 1,1,2,3,4)
- **Tiny version**: 7 cards (3 geishas, ranks 2,2,3)

### Game Setup
- **Medium version**: Start with 2 cards, 3 actions per player (6 turns total)
- **Tiny version**: Start with 1 card, 2 actions per player (4 turns total)

### Actions
- **Medium version**: RESERVE, DISCARD, SPLIT
- **Tiny version**: RESERVE, SPLIT (no DISCARD)

### Win Condition
- **Medium version**: Control 3+ geishas OR 6+ favor points
- **Tiny version**: Control 2+ geishas (simple majority)

## Quick Start

### Run a Test Game

```bash
python tiny_hana/tiny_hana.py
```

### Use in Code

```python
from tiny_hana import Game, RandomAgent

agent0 = RandomAgent()
agent1 = RandomAgent()

game = Game(agent0, agent1, verbose=True)
winner = game.play_game()

print(f"Winner: Player {winner}")
```

## Game Rules

### Setup
- **Cards**: 7 cards total across 3 suits (geishas)
  - GEISHA_1: rank 2 (2 cards)
  - GEISHA_2: rank 2 (2 cards)
  - GEISHA_3: rank 3 (3 cards)
- **Players**: 2 players competing for geisha favors
- **Starting hand**: Each player gets 1 card

### Round Structure
- Players alternate turns (4 turns total per round, 2 per player)
- Each turn: draw 1 card, then perform 1 action
- Each of the 2 action types can only be used once per player per round

### Actions

1. **RESERVE**: Take 1 card from hand and reserve it (hidden until end of round). You get this card.
2. **SPLIT**: Choose 2 cards from hand and reveal them. Opponent chooses one, you get the other.

### Winning Conditions

At the end of each round, favors are updated:
- For each geisha, the player with **strictly more** cards of that suit wins its favor
- If tied, favor remains where it was (neutral or with previous owner)

**Win Condition**: Control 2 or more geishas (simple majority)

If no winner, shuffle and start a new round with updated favors.

## Example Agent Implementation

```python
from tiny_hana import Agent, Action, ActionType, RoundInfoState

class GreedyAgent(Agent):
    """Prioritizes high-value geisha (GEISHA_3)."""

    def get_action_distribution(self, info_state: RoundInfoState):
        pending_split = info_state.get_pending_split()
        if pending_split is not None:
            choice = 0 if pending_split[0].suit.rank >= pending_split[1].suit.rank else 1
            return [(Action(ActionType.SPLIT_RESPONSE, [], choice=choice), 1.0)]

        hand, used_actions = info_state.get_current_hand_and_actions()

        # Sort by rank (GEISHA_3 is rank 3, others are rank 2)
        sorted_hand = sorted(hand, key=lambda c: c.suit.rank, reverse=True)

        # Reserve highest value card if possible
        if ActionType.RESERVE not in used_actions:
            return [(Action(ActionType.RESERVE, [sorted_hand[0]]), 1.0)]

        # Split with remaining cards
        if ActionType.SPLIT not in used_actions and len(hand) >= 2:
            return [(Action(ActionType.SPLIT, hand[:2]), 1.0)]

        # Fallback
        return [(Action(ActionType.RESERVE, [hand[0]]), 1.0)]
```

## Advantages of Tiny Version

1. **Faster games**: 4 turns vs 6 turns per round
2. **Simpler strategy space**: 2 actions vs 3 actions
3. **Easier to analyze**: Fewer cards and geishas
4. **Good for learning**: Simpler rules, same core mechanics
5. **Quick testing**: Faster iteration for AI development

## Code Structure

### Core Classes

- **`Suit`**: Enum for the 3 geishas with rank property
- **`Card`**: Represents a single card
- **`ActionType`**: Enum for RESERVE and SPLIT
- **`Action`**: Represents a player action
- **`PlayerState`**: Tracks one player's cards and used actions
- **`GameState`**: Manages the complete game state
- **`Agent`**: Abstract base class for AI agents
- **`RandomAgent`**: Example agent making random moves
- **`Game`**: Main game controller

## RoundInfoState API

When your agent's `select_action` method is called, `info_state` is a
`RoundInfoState` object. Use helper methods:

```python
pending_split = info_state.get_pending_split()
hand, used_actions = info_state.get_current_hand_and_actions()
my_collected, opponent_collected = info_state.get_collected_cards()
my_reserved = info_state.get_my_reserved()
favors = info_state.get_favors()
opponent_used_actions = info_state.get_opponent_used_actions()
turn = info_state.get_current_turn()
available_actions = info_state.get_available_action_types()
```

## Strategy Tips

1. **GEISHA_3 is valuable**: It has rank 3 (3 cards), so controlling it helps you win
2. **Track card distribution**: With only 7 cards, counting is easier
3. **Reserve wisely**: You only have 2 actions total
4. **SPLIT timing**: Since no DISCARD, SPLIT is your only way to share cards
5. **Favor persistence**: Like the full game, favors carry across rounds

## Notes

- **Partially observable**: Can't see opponent's hand or reserved cards
- **No DISCARD action**: Simpler information hiding
- **Simple win condition**: Just need 2 out of 3 geishas
- **Quick rounds**: Average game finishes in 1-2 rounds

## Comparison Table

| Feature | Full Hanamikoji | Tiny Hanamikoji |
|---------|----------------|-----------------|
| Total cards | 11 | 7 |
| Geishas | 5 | 3 |
| Geisha ranks | 1,1,2,3,4 | 2,2,3 |
| Starting hand | 2 cards | 1 card |
| Actions per player | 3 | 2 |
| Turns per round | 6 | 4 |
| Available actions | RESERVE, DISCARD, SPLIT | RESERVE, SPLIT |
| Win condition | 3+ geishas OR 6+ points | 2+ geishas |
| Average game length | ~3-5 rounds | ~1-2 rounds |
