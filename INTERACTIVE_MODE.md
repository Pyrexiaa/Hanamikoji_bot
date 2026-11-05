# Interactive Mode - Play Against AI

Play Hanamikoji directly in your terminal against AI opponents!

## Quick Start

```bash
python play_interactive.py
```

## Features

### Beautiful Text Interface
- Clean, bordered display showing all game information
- Real-time game state updates
- Clear action menus
- Visual card displays

### Full Game Information
The interface shows:
- **Your Status**: Hand, collected cards, reserved count, discarded cards
- **Opponent Status**: Hand size, collected cards, reserved count, revealed discards
- **Geisha Favors**: Current control status for each geisha
- **Actions Used**: Track which actions you and opponent have used
- **Round/Turn Info**: Always know where you are in the game

### Choose Your Variant and Opponent

The CLI first asks which variant to play:
- **Hanamikoji** (medium) — 5 geishas, 3 actions (RESERVE/DISCARD/SPLIT), 11 cards
- **Tiny Hanamikoji** — 3 geishas, 2 actions (RESERVE/SPLIT), 7 cards
- **Full Hanamikoji** — 7 geishas, 4 actions (RESERVE/TRADE-OFF/GIFT/COMPETITION), 21 cards (the published game)

Then choose an AI opponent. Hanamikoji and Tiny Hanamikoji offer:
1. **RandomAgent** - Makes random moves (easy)
2. **GreedyAgent** - Prioritizes high-value cards (medium)
3. **BalancedAgent** - Spreads influence across geishas (medium)
4. **CardCountingAgent** - Tracks revealed cards (hard)

Tiny Hanamikoji also adds:

5. **AdaptiveAgent** - Adjusts strategy based on game state (hard)

Full Hanamikoji offers RandomAgent, GreedyAgent and CardCountingAgent.

### Intuitive Controls

All actions are menu-driven with clear prompts:
- **RESERVE**: Choose 1 card to hide
- **DISCARD**: Choose 2 cards, then select which to reveal
- **SPLIT**: Choose 2 cards to offer opponent

## Example Session

```
==============================================================
  HANAMIKOJI - Interactive Game
==============================================================

Choose your opponent:
  [1] RandomAgent     - Makes random moves
  [2] GreedyAgent     - Prioritizes high-value cards
  [3] BalancedAgent   - Spreads across multiple geishas
  [4] CardCountingAgent - Tracks revealed cards

Select opponent [1-4]: 2

✓ You will play against: GreedyAgent

==============================================================
Game starting! You are Player 0.
==============================================================

╔══════════════════════════════════════════════════════════╗
║ ROUND 1  •  TURN 1/6                                     ║
╠══════════════════════════════════════════════════════════╣
║ YOUR STATUS:                                             ║
║   Collected: 0 cards                                     ║
║   Reserved: 0 card(s) (hidden)                           ║
╠══════════════════════════════════════════════════════════╣
║ OPPONENT STATUS:                                         ║
║   Hand size: 2 cards                                     ║
║   Collected: 0 cards                                     ║
║   Reserved: 0 card(s) (hidden)                           ║
╠══════════════════════════════════════════════════════════╣
║ GEISHA FAVORS:                                           ║
║   GEISHA_1     (rank 1): [---]                           ║
║   GEISHA_2     (rank 1): [---]                           ║
║   GEISHA_3     (rank 2): [---]                           ║
║   GEISHA_4     (rank 3): [---]                           ║
║   GEISHA_5     (rank 4): [---]                           ║
╠══════════════════════════════════════════════════════════╣
║ ACTIONS USED:                                            ║
║   You:      None                                         ║
║   Opponent: None                                         ║
╚══════════════════════════════════════════════════════════╝

============================================================
YOUR HAND:
============================================================
  [0] GEISHA_5 (rank 4)
  [1] GEISHA_3 (rank 2)
  [2] GEISHA_4 (rank 3)

AVAILABLE ACTIONS:
  [1] RESERVE: Reserve 1 card (hidden, you get it)
  [2] DISCARD: Discard 2 cards (reveal 1, hide 1)
  [3] SPLIT: Split 2 cards (opponent picks 1, you get the other)

Choose action [1-3]: 1

→ RESERVE: Choose a card to reserve (hidden)
  Card to reserve [0-2]: 0

  → Reserving: GEISHA_5
```

## Tips for Playing

### Strategy Tips
1. **RESERVE your best cards** - They're hidden from opponent
2. **Reveal low-value cards when DISCARDING** - Keep information hidden
3. **Make SPLIT difficult** - Offer cards of similar value
4. **Watch opponent's patterns** - Learn their strategy
5. **Track revealed cards** - Use card counting to your advantage

### Understanding the Display

**Favor Markers:**
- `[YOU]` - You control this geisha
- `[OPP]` - Opponent controls this geisha
- `[---]` - Neutral (no one controls)

**Win Conditions:**
- Control **3 or more geishas**, OR
- Control **6+ favor points**
- Points win takes priority!

### Common Actions

**When to RESERVE:**
- High-value cards (rank 3-4)
- Cards you need to win a specific geisha
- When you want to hide information

**When to DISCARD:**
- Low-value cards you don't need
- Cards where opponent has strong lead
- To remove cards from the game strategically

**When to SPLIT:**
- When you have two good cards and want one
- To force opponent into a difficult choice
- When both cards are valuable to you

## Keyboard Controls

- **Numbers (0-9)**: Select cards and actions
- **Enter**: Confirm selections
- **Ctrl+C**: Exit game at any time

## Error Handling

The interface provides clear error messages:
- Invalid card selection
- Action already used
- Not enough cards for action
- Invalid choices

Simply follow the prompts to correct any issues!

## Advanced Features

### Human Agent Implementation

The `HumanAgent` class can be used in your own code:

```python
from play_interactive import HumanAgent
from medium_hana import Game, GAME_CONFIG
from medium_hana.example_agents import GreedyAgent

# Create agents
human = HumanAgent("Player 1", game_config=GAME_CONFIG)
ai = GreedyAgent()

# Play game
game = Game(human, ai, verbose=False)
winner = game.play_game()

print(f"Winner: Player {winner}")
```

### Custom Opponents

You can easily add your own AI agents to the opponent selection:

```python
# In play_interactive.py, modify select_opponent():
opponents = {
    1: ("RandomAgent", RandomAgent()),
    2: ("GreedyAgent", GreedyAgent()),
    3: ("MyCustomAgent", MyCustomAgent()),  # Add your agent here
}
```

## Troubleshooting

**Problem**: Display looks messy
- **Solution**: Use a terminal with at least 80 character width

**Problem**: Can't see full game state
- **Solution**: Increase terminal height or scroll up

**Problem**: Input doesn't work
- **Solution**: Make sure terminal supports standard input (not all IDEs do)

**Problem**: Want to quit
- **Solution**: Press Ctrl+C at any time

## Educational Use

### For Students
- **Learn by playing**: Understand game mechanics hands-on
- **Test strategies**: See what works against different AI
- **Debug your agent**: Play against your own AI to find weaknesses
- **Analyze opponent**: Watch AI patterns to improve your own agent

### For Instructors
- **Demonstration**: Show game mechanics in class
- **Testing**: Verify AI agents work correctly
- **Comparison**: Let students play against reference implementations
- **Engagement**: Make the assignment more interactive and fun!

## Example Play Session

Here's what a typical turn looks like:

```
YOUR HAND:
  [0] GEISHA_3 (rank 2)
  [1] GEISHA_5 (rank 4)

AVAILABLE ACTIONS:
  [1] RESERVE: Reserve 1 card (hidden, you get it)
  [2] DISCARD: Discard 2 cards (reveal 1, hide 1)

Choose action [1-2]: 2

→ DISCARD: Choose 2 cards to discard
  (You'll then choose which one to reveal to opponent)

  First card [0-1]: 0

  Remaining cards:
    [1] GEISHA_5 (rank 4)

  Second card [0-1]: 1

  Cards to discard:
    [0] GEISHA_3 (rank 2)
    [1] GEISHA_5 (rank 4)

  Which card to REVEAL to opponent? [0-1]: 0

  → Revealing: GEISHA_3
  → Hiding: GEISHA_5
```

## Have Fun!

Enjoy playing Hanamikoji against the AI. May the best strategist win! 🎴

## Web GUI (Browser-Based)

A fully-featured web interface is also available as an alternative to the CLI:

```bash
python gui/server.py
```

Then open `http://localhost:8080` in your browser. It supports all three variants, all agents
(including the shared heuristic agents), a favor-setup screen for single-round mode, and
a visual event log.
