# Hanamikoji Quick Reference Card

Quick reference for implementing agents. Keep this handy while coding!

> The examples use medium_hana (5 geishas; RESERVE, DISCARD and SPLIT). The tasks use tiny_hana
> (3 geishas; RESERVE and SPLIT only): the API is the same, so swap `medium_hana` for `tiny_hana`.

## Game Overview

- **Goal**: Control 3+ geishas OR 6+ favor points
- **Cards**: 11 total across 5 suits (ranks: 1, 1, 2, 3, 4)
- **Each Round**: 3 actions per player, 5 cards seen
- **Win Condition**: Most cards of a suit wins that geisha's favor

## The Three Actions

| Action | Cards Used | Effect |
|--------|------------|--------|
| **RESERVE** | 1 | Keep card hidden, get it at round end |
| **DISCARD** | 2 | Remove 2 cards from game; reveal ONE to opponent, hide the other |
| **SPLIT** | 2 | Opponent picks 1, you get the other |

*Each action can only be used ONCE per round*

### Full Hanamikoji (full_hana)

7 geishas (ranks 2, 2, 2, 3, 3, 4, 5; 21 cards), 6-card hands, 4 actions each.
Win with 11+ points (checked first) or 4+ geishas.

| Action | Cards Used | Effect |
|--------|------------|--------|
| **RESERVE** | 1 | The "secret": keep card hidden, get it at round end |
| **TRADE_OFF** | 2 | Discard both face down; the opponent sees neither |
| **GIFT** | 3 | Opponent takes 1 (`GIFT_RESPONSE`, choice 0-2); you keep 2 |
| **COMPETITION** | 4 | Two pairs, `cards[0:2]` and `cards[2:4]`; opponent takes one (`COMPETITION_RESPONSE`, choice 0-1) |

## RoundInfoState API

```python
pending_split = info_state.get_pending_split()       # SPLIT only: [card0, card1] or None
pending_offer = info_state.get_pending_offer()       # any offer: (ActionType, cards) or None
hand, used_actions = info_state.get_current_hand_and_actions()
my_collected, opponent_collected = info_state.get_collected_cards()
my_reserved = info_state.get_my_reserved()
my_revealed, my_hidden, opponent_revealed = info_state.get_discarded_cards()
opponent_used_actions = info_state.get_opponent_used_actions()
favors = info_state.get_favors()
turn = info_state.get_current_turn()
available_actions = info_state.get_available_action_types()

key = str(info_state)                  # readable information-set key (policy tables)
compact = info_state.compact_key()     # same information as bytes, ~7x shorter
# engine.compact_key_to_string(compact) == key; engine.string_to_compact_key(key) == compact
```

## Saving a Strategy (Policy Tables)

```python
table = {str(info_state): {action: prob, ...}, ...}   # may be partial
PolicyTableAgent(table).dump_artifact("s.json", variant="medium_hana")         # readable JSON
PolicyTableAgent(table).dump_binary_artifact("s.bin", variant="medium_hana")   # binary, smaller
CompactPolicyTableAgent(table).dump_binary_artifact("c.bin", variant="medium_hana")  # compact keys

agent = PolicyTableAgent.from_artifact("s.bin", Suit, fallback=RandomAgent())  # JSON or binary
# Without fallback=..., an information set missing from the table raises ValueError.
# C++: engine_cpp/include/hana/policy_table.h writes the same files.
```

## Creating Actions

```python
# RESERVE: Take 1 card
Action(ActionType.RESERVE, [card])

# DISCARD: Discard 2 cards, reveal one (choice=0 reveals card1, choice=1 reveals card2)
Action(ActionType.DISCARD, [card1, card2], choice=0)  # Reveals card1, hides card2

# SPLIT: Offer 2 cards, opponent chooses
Action(ActionType.SPLIT, [card1, card2])

# SPLIT_RESPONSE: Required when opponent has an unresolved split
Action(ActionType.SPLIT_RESPONSE, [], choice=0)  # Take first offered card

# Full Hanamikoji
Action(ActionType.TRADE_OFF, [card1, card2])
Action(ActionType.GIFT, [card1, card2, card3])
Action(ActionType.COMPETITION, [a1, a2, b1, b2])          # pairs (a1, a2) and (b1, b2)
Action(ActionType.GIFT_RESPONSE, [], choice=2)            # take the third offered card
Action(ActionType.COMPETITION_RESPONSE, [], choice=1)     # take the second pair
```

## Common Code Patterns

### Checking Available Actions
```python
hand, used_actions = info_state.get_current_hand_and_actions()
if ActionType.RESERVE not in used_actions:
    # Can still use RESERVE action
```

### Sorting Cards by Value
```python
sorted_hand = sorted(hand, key=lambda c: c.suit.rank, reverse=True)
highest_card = sorted_hand[0]  # Highest rank
lowest_card = sorted_hand[-1]  # Lowest rank
```

### Counting Collected Cards
```python
from collections import defaultdict

my_suits = defaultdict(int)
my_collected, _ = info_state.get_collected_cards()
for card in my_collected:
    my_suits[card.suit] += 1

# Now: my_suits[Suit.GEISHA_5] = number of GEISHA_5 cards you have
```

### Checking Which Geishas You're Winning
```python
favors = info_state.get_favors()
for suit in Suit:
    if favors[suit] == -1:
        print(f"I control {suit.name}")
```

### Card Counting
```python
# Total cards in game per suit
TOTAL_CARDS = {
    Suit.GEISHA_1: 1,
    Suit.GEISHA_2: 1,
    Suit.GEISHA_3: 2,
    Suit.GEISHA_4: 3,
    Suit.GEISHA_5: 4,
}

# Track what you've seen
hand, _ = info_state.get_current_hand_and_actions()
my_collected, opponent_collected = info_state.get_collected_cards()
seen = hand + my_collected + opponent_collected
seen_counts = defaultdict(int)
for card in seen:
    seen_counts[card.suit] += 1

# Calculate unseen cards
unseen = {suit: TOTAL_CARDS[suit] - seen_counts[suit] for suit in Suit}
```

## Strategy Tips

### Basic Strategies
1. **Greedy**: Reserve high-value geishas, discard low-value
2. **Balanced**: Spread across multiple geishas
3. **Focused**: Try to win 3 specific geishas

### Intermediate Strategies
1. **Card Counting**: Track revealed cards to predict opponent's hand
2. **Denial**: Discard/reserve cards opponent needs
3. **Split Optimization**: Make opponent choose between equally bad options

### Advanced Strategies
1. **Minimax**: Look ahead several moves
2. **Probabilistic**: Calculate expected values based on unseen cards
3. **Opponent Modeling**: Adapt strategy based on opponent's pattern

## Testing Your Agent

### Single Game (Detailed)
```python
from medium_hana import Game
from medium_hana.example_agents import GreedyAgent

my_agent = MyAgent()
opponent = GreedyAgent()
game = Game(my_agent, opponent, verbose=True)
winner = game.play_game()
```

### Multiple Games (Win Rate)
```python
wins = 0
for _ in range(100):
    my_agent = MyAgent()
    opponent = GreedyAgent()
    game = Game(my_agent, opponent, verbose=False)
    if game.play_game() == 0:  # You are player 0
        wins += 1
print(f"Win rate: {wins}%")
```

### Tournament
```python
from tournament import AgentEvaluator

evaluator = AgentEvaluator()
result = evaluator.play_match(MyAgent, GreedyAgent, num_games=100)
evaluator.print_match_results(result)
```

## Common Bugs

### Bug: "Invalid action selected"
**Cause**: Action validation failed
**Fix**:
- Check if action already used
- Verify cards are actually in hand
- Ensure correct number of cards (RESERVE=1, DISCARD=2, SPLIT=2)

### Bug: "List index out of range"
**Cause**: Trying to access card that doesn't exist
**Fix**:
- Check `len(hand)` before accessing indices
- Make sure you have enough cards for the action

### Bug: Agent always uses same strategy
**Cause**: Not adapting to game state
**Fix**:
- Use observation data to make decisions
- Check favor state, opponent's cards, etc.

### Bug: Poor performance
**Cause**: Strategy not considering important factors
**Fix**:
- Add logging to see what agent is doing
- Compare decisions to example agents
- Test individual components of strategy

## Debugging Tips

### Print Current State
```python
def get_action_distribution(self, info_state: RoundInfoState):
    hand, used_actions = info_state.get_current_hand_and_actions()
    print(f"Round: {info_state.round_number}, Turn: {info_state.get_current_turn()}")
    print(f"Hand: {hand}")
    print(f"Used actions: {used_actions}")
    # ... your logic
    return {Action(ActionType.RESERVE, [hand[0]]): 1.0}   # action -> probability
```

### Validate Choices
```python
# Before returning action, check it
action = Action(ActionType.RESERVE, [best_card])
if action.validate(hand, used_actions):
    return action
else:
    print("ERROR: Invalid action!")
    # Return fallback action
```

### Track Decision Reasoning
```python
# Add comments or logs explaining WHY you chose this action
print(f"Reserving {card} because it's highest value and we need it")
```

## Getting Help

1. Read error messages carefully
2. Use `verbose=True` to watch game play
3. Print `RoundInfoState` helper outputs to understand state
4. Compare your code to example agents
5. Test incrementally (one feature at a time)

## Remember

- Each action can only be used ONCE per round
- You see 5 cards total per round (start with 2, draw 3)
- Ties maintain previous favor (not always neutral!)
- 6+ points wins even if opponent has 3 geishas
- Card counting is powerful - track what's been played!

---

Good luck building your AI! 🎴
