# Introduction

In this project, we will create bot(s) to play the board/card game [Hanamikoji](https://en.boardgamearena.com/gamepanel?game=hanamikoji). It is a two-player zero-sum (2p0s) game that involves deception and bluffing.

We will guide you through a few basic tasks, and at the end, you should have a fully functional bot for a tiny variant of the game. After which, you are free to choose which direction you take the project in. 

## Goals of the project

- To appreciate the importance and complexities brought about by imperfect information.
- To gain experience in large-scale game solving by navigating a reasonably large game tree alongside information sets.
- To provide a chance to concretely experiment with some concepts relevant to game-solving, including Nash equilibrium (NE), exploitability, saddle-point residual/gap, perfect recall, the sequence form representation.
- To better understand algorithms to solve 2p0s games, e.g., counterfactual regret minimization, double oracle/PSRO, value iteration.
- Have a chance to implement tweaks/improvements outside the syllabus that will speed up game-solving.

## What this project is *not* about

- Finding a perfect bot. Yes, we will try to create strong bots, but if we were truly serious about optimal ones, we will probably not be using python for both the game engine and solver; exact solvers are computationally intensive and would typically be written in a compiled language with parallelization, possibly on a supercomputer. 
- I am not here to grade you on the amount of compute you have access to. Neither do I wish for you to spend time porting your code/the engine to C/C++.
- I am less interested in absolute performance of your bot as compared to you demonstrating that you have learnt something in class, and preferably beyond that. 
- I am not at all impressed by students who run experiment after experiment for the sole purpose of tuning hyperparameters --- quantity is not quality. Far more impressive is the selective, careful choice of *what* experiments to run and *why* you chose to run them.

## Grading Scheme
- 60% of the points will be related to the guided tasks (Tasks 1–3) and your final bot on the tiny task (your Task 3 strategy).
- 40% on extensions, alongside a written report. The report contains what methods you have tried, any interesting observations, what you did to verify correctness, experimental results regarding runtime, convergence rates etc. The report should be a maximum of 5 pages, A4, 12 pt font, 2.5 cm margins on all sides.


As long as you complete the basic tasks correctly and complete a bot that solves the game on the smallest scale, you will obtain most of the first 60% or the points.

## What to submit
Submit **one zip file on Canvas** (any file name). Your work is marked by an autograder, so the zip must
have exactly this layout. The paths are case-sensitive and sit at the top level of the zip, not inside an
enclosing folder:

```
student_attempts/        all your code for Tasks 1–3, including the provided files
    utility.py           Task 1: compute_posterior_given_opponent_agent
    solver.py            Task 2: compute_best_response and saddle_point_residual; Task 3: your solver
    ...                  any other files your code imports
tasks/nash_p1.npz        Task 3: Player 1's strategy
tasks/nash_p2.npz        Task 3: Player 2's strategy
report.pdf               Task 4: your report (PDF, at most 5 pages)
task4/                   Task 4: the code and notebooks for your experiments
```

**Build the zip with `python make_submission.py`** (from the project root). It collects exactly these
paths from your project (put the report at `report.pdf` and your Task 4 work in `task4/` first),
checks them, and writes `submission.zip` for you to upload. It stops with an error if your code cannot
be imported or a strategy file is malformed, and warns about anything missing or not yet implemented.

How each part is assessed, and what that requires:

| Task | Assessed on | Requirements |
|------|-------------|--------------|
| 1 | `compute_posterior_given_opponent_agent` in `utility.py`, run on hidden test cases | Keep the file name, the function name and its arguments as in the starter code. |
| 2 | `compute_best_response` and `saddle_point_residual` in `solver.py`, run on hidden test cases | As for Task 1. |
| 3 | The strategy in `tasks/nash_p1.npz` and `tasks/nash_p2.npz`: how exploitable it is (its saddle-point residual). Your solver code is submitted too, but not marked. | Solve the default `TreeplexGame()` (one round of tiny_hana from neutral favors) and save with `TreeplexGame.save_agent`, as the last code cell of `tasks/task3.ipynb` does. Keep `treeplex_representation.py` unchanged: the strategies are read with the provided one. |
| 4 | `report.pdf` | The code in `task4/` supports the report; it is not run. |

The autograder imports `utility.py` and `solver.py` with only the packages in `requirements.txt` (and the
Python standard library) installed, using the Python engine. If your code needs another package, such as
SciPy for a linear program, import it inside the function that uses it, not at the top of the file. Your
code must not depend on files outside the zip.

## Disclaimer
This is the first time this project was created, and the game engine was partially vibe coded. Much effort has been put in to ensure that there are no bugs, but mistakes sometimes occur. Please inform the lecturer if there are any errors. We apologize in advance if there are any.

# Rules of the Game

Though we will be working with much smaller variants, you are highly encouraged to (i) try the game and/or (ii) read the rules of the game in the link provided above to get a feel of how the game is played. 

The version of the game we will mostly use is *tiny_hanamikoji*, abbreviated as *tiny_hana*. There is a medium-sized version that I have included, but solving that is not part of the main project but rather an potential extension.

A GUI and interative CLI interface has been included. It is worth spending some time playing against some custom (and weak) bots to see how the game is played.

## Getting started

> **Note:** Run all commands below from the **repo root directory** (the folder that contains `tournament.py`, `README.md`, etc.).

### Setup
You need Python 3.9 or newer (tested with 3.9 and 3.14). Install the packages with:
```bash
pip install -r requirements.txt
```

### Play a game with GUI
```bash
python gui/server.py
```

Then open `http://localhost:8080`. Pick a variant, an opponent and whether to play a full
game or a single round. To move, click one of your action tokens, then the cards in your
hand, then Confirm (you can also pick the cards first); answer the AI's offers by clicking
the card or pair you want. Keys: `1`–`4` choose an action, `Enter` confirms, `Esc` cancels.
We recommend this to build some intuition as to how to play the game.

### Play a game via CLI
```bash
python play_interactive.py
```

Play against AI opponents in your terminal with text interface.

See [INTERACTIVE_MODE.md](INTERACTIVE_MODE.md) for full details.

### Run a Tournament

```bash
python tournament.py                    # medium_hana; add --variant tiny or --variant full
```

Watch AI agents compete against each other. `--policy-table FILE` adds an agent saved as a policy table (Task 4).

### Work on the tasks

The tasks are Jupyter notebooks in `tasks/`. If Jupyter is not installed, `pip install notebook`; then start it
from the repo root with `jupyter notebook` and open `tasks/task1.ipynb`. Each notebook's first code cell sets up
the Python path, so run the cells from the top.

The tasks run on the Python engine, which is the default. A faster C++ engine with the same API runs them too:
[CPP_ENGINE.md](CPP_ENGINE.md) explains when it helps, how to build and switch to it, and how to port code
written on the Python engine.

### Run a Quick Demo

```bash
python demo.py
```

Run a short example game between two baseline agents.

## The basic rules

There are $n$ Geisha that 2 players are trying to win the favors of. There are $n$ types of items, one type for each Geisha, and each player jostles for their favors by obtaining items that are associated with them. The items are represented by cards. How items are obtained and distributed between players is dependent on the players' actions taken.

For tiny_hana, $n=3$. Geisha 1 and 2 have 2 items associated to them, and Geisha 3 has 3 items, making for a total of $2+2+3=7$ cards. For medium_hana, there are 5 Geishas with $1+1+2+3+4=11$ items. full_hana is the published game: 7 Geishas with $2+2+2+3+3+4+5=21$ items (see [Full Hanamikoji](#full-hanamikoji-full_hana) below).

_Remark_: The terms "Geisha" and "favors" are what was used in the original game. This has no bearing on the strategic aspect of the game, which is the focus of the class. If you prefer, genders can be swapped, or one could replace "Gesiha" with "Pets", "Professors", "Pokemon", or whatever suits your taste.

The game itself is played in one of more rounds, at the end of each round, if there is no winner, then we start a new round with some small changes in the starting position (this will be clear later). 

### Anatomy of a Round
At the beginning of each round, each geisha is either (i) neutral, (ii) player 1 favored, or (iii) player 2 favored. Which way they are favored depends on the outcome from the previous round; if it is the first round then all geisha are neutral.

At the start of the round, cards are shuffled in a deck. Each players is dealt a number of cards to begin with, for tiny_hana, this is just a single card. These cards are in the player's hand and private to themselves. 

Each round is played in $m$ *turns*, taken alternately between players. At the start of each turn, the player-to-move first draws a card randomly from a pile (the card drawn is private). He then takes an *action* from a list. Each action can only be taken exactly once per round (there are $m$ actions in total). For tiny_hana, the actions are the following:
- *Reserve*. The player-to-move picks one card from his hand (privately) and places it face down (i.e., it is no longer in his hand). This item will be considered as belonging to the player at the end of the round.
- *Split*. The player-to-move picks 2 cards from his hand and reveals them to his opponent. His opponent chooses one of them (publicly) and gets that item. The player-to-move gets the remaining item. Note that for tiny_hana, there can be at most 2 cards in the player's hand, so there is no real choice as to which 2 cards to reveal.
- *Discard*. For medium_hana _only_. Select two cards and discard them (no one owns them). The identity of one of them is revealed to the opponent and the other is kept private.

Once both players have played both (all) their actions, the round is over. In tiny_hana, each player will publicaly own 2 cards and have reserved a single card. There will also be exactly one undrawn card remaining in the deck whose identity is not revealed to either player. We now move on to update geisha favors. For each geisha, we see if who has more items associated to it. Make sure to include the item that you chose to reserve. The player who has strictly more items wins that geisha's favor. If there is a tie, the favor remains the same as it was at the start of the round (which could still be neutral, based on the outcome of the previous round). 

After favors are updated, we check if a player has won. For tiny_hana, a player wins if it has the favor of 2 or more geisha (i.e., the majority).  

_Remark_: The winning conditions are slightly more complicated for medium_hana; a player wins if it wins the strict majority (3/5 or more geisha) or scores 6 or more points, where the value of each geisha is equal to how many items it has (e.g., winning the favor of the gesha with 4 items scores 4 points), the latter condition takes priority if both conditions are satisfied by different players.

For the purposes of this game, a player who wins gets a payoff of +1, the loser gets a payoff of -1. The game is over. However, if there is no winner, we collect all the cards, shuffle them again and restart a new round. The starting favors in the new round are based on the *updated favors* in the current round. So, if I win the favor of Geisha 1 and the remaining 2 Geisha remain neutral, then I will have a huge advantage in the following round (it turns out this particular outcome is not possible, but you get the point).

In the multi-round verison of the game, we will rounds repeat over and over until a winner is eventually declared.

### Full Hanamikoji (full_hana)
full_hana follows the published rules. Each player is dealt 6 cards, draws one at the start of each of their 4 turns, and must use each of these 4 actions exactly once per round:
- *Reserve* (the "secret" action). As above: one card kept face down, counted at the end of the round.
- *Trade-off*. Discard two cards face down. They take no part in the round, and the opponent only learns that a trade-off happened.
- *Gift*. Offer three cards face up. The opponent takes one; you keep the other two.
- *Competition*. Offer two pairs face up. The opponent takes one pair; you get the other.

A player wins with 11 or more points (checked first) or with the favor of 4 or more Geisha. One card stays undealt each round, as in the physical game.

The game tree of full_hana is far too large to enumerate, so the exhaustive tools (treeplex extraction, CFR, the Markov solver) refuse it. Playing, sampling and agents work as for the other variants. The C++ engine (`HANA_ENGINE=cpp`, see [CPP_ENGINE.md](CPP_ENGINE.md)) supports all three variants.

## For this project
For this project, we introduce two variants of the game. 

- The **single-round** game. Here, you start off with a given favor state (public knowledge) and play a single round. If there is no winner at the end of the round, the game is a tie and both players get $0$.
- The **multiple-round/markov** game. Here, after ties, we start a new round. This continues until someone eventually wins.

Both versions are compatible with tiny_hana, medium_hana and full_hana (for full_hana, play and sampling only; see above).

# Repository Structure

```
hana/
├── README.md                # Start here — overview, rules, and project structure
├── CPP_ENGINE.md            # Using the faster C++ engine, and porting code to it
├── make_submission.py       # Builds and checks your submission zip (see "What to submit")
├── engine/                  # Shared game engine (used by all variants)
│   ├── constants.py         #   CHANCE_PLAYER, PLAYER_1, PLAYER_2
│   ├── enums.py             #   ActionType, ObservationType
│   ├── models.py            #   Card, Action, GameConfig dataclasses
│   ├── events.py            #   ObservationEvent subclasses
│   ├── info_state.py        #   RoundInfoState — your information interface
│   ├── state.py             #   GameState, PlayerState — core game state
│   ├── agents.py            #   Agent base class, RandomAgent
│   ├── game.py              #   Game runner (multi-round play)
│   ├── policy_table.py      #   PolicyTableAgent — save/load agents as JSON or binary
│   ├── binary_policy_table.py #  the binary policy-table format (large tables)
│   └── *.py                 #   (deck.py, legal_actions.py, render.py — internal helpers)
├── tiny_hana/               # Tiny variant (3 geishas, 7 cards, 2 actions per player)
│   ├── tiny_hana.py         #   Game engine entry point
│   └── example_agents.py    #   GreedyAgent, BalancedAgent, CardCountingAgent, AdaptiveAgent
├── medium_hana/             # Medium variant (5 geishas, adds DISCARD action)
│   ├── medium_hana.py
│   └── example_agents.py
├── full_hana/               # Full Hanamikoji (7 geishas, 21 cards; trade-off, gift, competition)
│   ├── full_hana.py
│   └── example_agents.py    #   GreedyAgent, CardCountingAgent
├── student_attempts/        # Your implementation folder
│   ├── env_config.py        #   Environment imports and player constants
│   ├── solver.py            #   Solver base class + CFRSolver stub
│   ├── treeplex_representation.py  # Treeplex and TreeplexGame
│   └── utility.py          #   Helper functions (posterior, payoffs)
├── tasks/                   # Guided task notebooks
│   ├── task1.ipynb
│   ├── task2.ipynb
│   ├── task3.ipynb
│   └── task4.ipynb
├── gui/                     # Web-based GUI server
│   └── server.py
├── play_interactive.py      # CLI interactive play
├── tournament.py            # Head-to-head tournament runner
├── QUICK_REFERENCE.md       # One-page summary of the engine API, with code patterns
├── engine_samples.py        # Runnable, annotated tour of the engine API
├── cpp_samples.cpp          # The same tour in C++, using the C++ engine directly
└── requirements.txt         # Python packages (pip install -r requirements.txt)
```

**Your implementation folder is `student_attempts` in the repo root.** There is some starter code there; you will complete functions and potentially add new files to finish the project. Run scripts from the **repo root**; the notebooks in `tasks/` run from `tasks/` and set up the path in their first code cell.

# The Agent Interface

Every agent inherits from `Agent` (`engine/agents.py`) and implements a single method:

```python
class Agent(ABC):
    @abstractmethod
    def get_action_distribution(self, info_state: RoundInfoState) -> Dict[Action, float]:
        """Return a mixed strategy as {action: probability, ...}.

        - Deterministic agent: return {chosen_action: 1.0}
        - Mixed strategy: all probabilities must sum to 1.0
        """
```

The engine calls `get_action_distribution` at every player decision node and samples an action from the returned distribution.

`RandomAgent` (also in `engine/agents.py`) shows the minimal implementation — it queries `info_state.get_legal_actions()` and assigns uniform probability to each action.

All reference agents for tiny_hana are in `tiny_hana/example_agents.py`:  
`GreedyAgent`, `BalancedAgent`, `CardCountingAgent`, `AdaptiveAgent`.

Read `engine/agents.py` for the source of both classes.

## Example agents (`tiny_hana/example_agents.py`)

For completeness, we describe four of the hand-crafted agents (actually, vibe-coded) agents that are provided as starting references. The contents of this cell are not essential for the completion of the project, so students may skip it. We are showcasing how a **heuristic** agent may be developed. As we will see, apart from knowing how to traverse the game tree, our game solving algorithms do not have to worry about such internals.

| Agent | Core idea |
|-------|-----------|
| `GreedyAgent` | Always reserves and targets the highest-rank geisha |
| `BalancedAgent` | Spreads influence — reserves the card for the suit it has least of |
| `CardCountingAgent` | Tracks visible cards; prioritises suits with fewer cards seen so far |
| `AdaptiveAgent` | Switches between defensive and aggressive play based on the current favor state |

All four are **deterministic** — `get_action_distribution` always returns `{chosen_action: 1.0}`.
A mixed-strategy agent would instead return a proper probability distribution over actions.

### `BalancedAgent` walkthrough

`BalancedAgent` is a good first example of how to use the `RoundInfoState` API:

```python
def _select_action(self, info_state: RoundInfoState) -> Action:
    # 1. Handle the split-response case first (opponent has offered two cards).
    pending_split = info_state.get_pending_split()
    if pending_split is not None:
        return self._select_split_response(info_state, pending_split)

    # 2. Read the current hand and which of our own actions are already used.
    hand, used_actions = info_state.get_current_hand_and_actions()

    # 3. Count cards per suit already secured (collected + reserved).
    my_collected, _ = info_state.get_collected_cards()
    my_reserved     = info_state.get_my_reserved()
    my_counts = {suit: 0 for suit in Suit}
    for card in my_collected + my_reserved:
        my_counts[card.suit] += 1

    # 4. Sort hand so the suit we have fewest of comes first.
    hand_by_need = sorted(hand, key=lambda c: my_counts[c.suit])

    # 5. Reserve the card we need most; then split with whatever remains.
    if ActionType.RESERVE not in used_actions:
        return Action(ActionType.RESERVE, [hand_by_need[0]])
    if ActionType.SPLIT not in used_actions:
        return Action(ActionType.SPLIT, hand[:2])
```

The split-response helper applies the same logic — pick the card for the suit
we currently have fewer of:

```python
def _select_split_response(self, info_state, cards):
    # Re-count secured cards, then pick whichever offered card fills the bigger gap.
    ...
    choice = 0 if my_counts[cards[0].suit] <= my_counts[cards[1].suit] else 1
    return Action(ActionType.SPLIT_RESPONSE, [], choice=choice)
```

Notice that the agent never sees the opponent's hand or the undrawn deck card —
everything is queried through `RoundInfoState` alone.

# Running a Game

The code below runs a full game between two `RandomAgent`s with `verbose=True`. Run it from the repo root and read through the output to understand:

```python

import sys, os
sys.path.insert(0, os.path.abspath('.'))

from tiny_hana.tiny_hana import Game, RandomAgent

game = Game(RandomAgent(), RandomAgent(), verbose=True)
winner = game.play_game()

if winner == 0:
    print("\nPlayer 1 wins.")
elif winner == 1:
    print("\nPlayer 2 wins.")
else:
    print("\nDraw.")
```

- How turns alternate between players
- What cards are drawn at each turn
- How the RESERVE and SPLIT actions are resolved
- How geisha favors update at the end of the round

_Remark_: **`sys.path.insert(0, '.')`** adds the repo root to Python's module search path so that `tiny_hana`, `engine`, and other packages are importable.

# The `GameState` API

`GameState` is the **mutable, full-information** game state used by algorithms and the engine
itself. Unlike `RoundInfoState` — which is an immutable, partial-information snapshot given
to agents — `GameState` sees everything and can be stepped forward or rewound.

For runnable examples of this API (playing a game step by step, chance outcomes, `undo_move`,
`to_key`), run `python engine_samples.py` and read its source. [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
summarises the whole API on one page.

## Node types

The `current_player` property is the single indicator of what kind of node you are at:

| `current_player` | Node type | What to call |
|-----------------|-----------|---------------|
| `CHANCE_PLAYER` (`-1`) | Chance node | `get_chance_actions_with_probs()` |
| `0` or `1` | Player decision node | `get_legal_actions()` |
| `None` | Terminal (round complete) | `terminal_payoff()` |

## Key methods

| Method | Returns | Notes |
|--------|---------|-------|
| `execute_action(action)` | `bool` | Apply an action in-place; returns `False` if illegal |
| `undo_move()` | — | Reverse the last `execute_action()` call in O(1) |
| `get_legal_actions()` | `Generator[Action]` | All legal actions at the current node (chance or player) |
| `get_chance_actions_with_probs()` | `Generator[(Action, float)]` | Chance actions with exact probabilities |
| `get_info_state(player_id)` | `RoundInfoState` | The partial-info view for that player at this node |
| `is_round_complete()` | `bool` | `True` once all turns have been played |
| `terminal_payoff(player_id)` | `float` | `+1 / -1 / 0` utility at a completed round |
| `to_key()` | `GameStateKey` | Hashable snapshot — use as a dict/set key |

## Tree traversal pattern

`execute_action` / `undo_move` modify the state in-place in O(1), making recursive
tree traversal fast without any copying:

```python
state.execute_action(action)
try:
    recurse(state)   # state is now one step deeper
finally:
    state.undo_move()  # always restored, even if recurse() raises
```

This pattern is used throughout Tasks 1–3 for game-tree traversal, payoff computation,
and solver algorithms.

## `get_info_state()` — the bridge between state and agent

`get_info_state(player_id)` is the most important method for algorithm design. It extracts
the partial-information view that a player would have at the current node, directly from
the full game state. Key uses:

- **Information set identification** — two nodes with the same `RoundInfoState` are
  indistinguishable to that player. This is the definition of an information set, and
  equality of `RoundInfoState` objects can be used to group nodes together in solvers.
- **Querying agent decisions** — call `agent.get_action_distribution(state.get_info_state(player_id))`
  to ask an agent what it would do at the current node, without the agent ever seeing
  the full state.
- **Computing posteriors** — given a target `RoundInfoState`, you can traverse the tree
  and collect all `GameState` nodes where `get_info_state(player_id)` matches, weighted
  by reach probability (Task 1).
- **Building the treeplex** — the treeplex groups nodes by their info state to construct
  the sequence-form strategy space (Tasks 2–3).

Information sets are keyed by `str(info_state)`, a readable string such as
`p0r1|ROUND_START@t0(hand=[0],...)|DRAW_CARD@t0(card=0)|...`; policy-table artifacts
use it too. Solvers that store millions of information sets can use
`info_state.compact_key()` instead: the same information as a few bytes (about 7×
shorter on medium_hana). `compact_key_to_string` and `string_to_compact_key` (from
`engine`) convert between the two exactly.

A strategy is saved as a **policy table**: information-set key → action
probabilities, with the actions written out (not as positions in a list), loaded with
`PolicyTableAgent.from_artifact(path, Suit)`. There are two file formats with the same
content: a readable JSON artifact (`dump_artifact`), and a binary file
(`dump_binary_artifact`) for tables with millions of entries, which can also use
compact keys (`CompactPolicyTableAgent`). `from_artifact` reads either. A table may
be partial: at an information set it does not list, the agent plays its `fallback`
agent, or raises if it has none.

Solvers written in C++ can write both formats with `engine_cpp/include/hana/policy_table.h`
(`write_policy_table_json`, `BinaryPolicyTableWriter`), keyed by
`RoundInfoState::to_string()` or `compact_key()`.

# The `RoundInfoState` API

`RoundInfoState` is the interface between the game engine and your **agent** at decision time. It is immutable — you cannot modify it, only query it. All information available to a playing agent must come through its helper methods. Solvers and algorithms, by contrast, operate on the full `GameState` and use `get_info_state()` to extract the agent's view at any node.

| Method | Returns | Notes |
|--------|---------|-------|
| `get_current_hand_and_actions()` | `(List[Card], Set[ActionType])` | Your private hand + which of your own action types are already used this round |
| `get_favors()` | `Dict[Suit, int]` | `−1` = Player 1 (id 0) holds the favor, `+1` = Player 2 (id 1), `0` = neutral; the same for both players' views |
| `get_collected_cards()` | `(List[Card], List[Card])` | Cards you won vs. cards opponent won from SPLIT exchanges |
| `get_my_reserved()` | `List[Card]` | Cards you have reserved (private to you) |
| `get_opponent_reserved_count()` | `int` | How many cards opponent reserved (count only — suit is hidden) |
| `get_pending_split()` | `Optional[List[Card]]` | The two cards offered when you must respond to a SPLIT; `None` otherwise |
| `get_opponent_used_actions()` | `Set[ActionType]` | Action types the opponent has already played this round |
| `get_legal_actions()` | `Generator[Action, ...]` | All legal actions at this decision node (deduplicated) |
| `get_available_action_types()` | `FrozenSet[ActionType]` | Variant capability set (RESERVE, SPLIT for tiny_hana; adds DISCARD for medium_hana) |
| `get_discarded_cards()` | `(my_rev, my_hid, opp_rev)` | medium_hana DISCARD action only |

**Hidden information:** your opponent's hand cards, which specific card they reserved, and the one undrawn deck card. Everything else is either public or derivable from the event log inside `RoundInfoState`.

**Action types in tiny_hana:** `RESERVE` (secure one card face-down) and `SPLIT` (offer two cards, opponent picks one). Each action type can be used **exactly once** per round per player.

# Overview of Tasks

All tasks are compulsory. Tasks 1, 2, and 3 are more guided. Task 4 is open-ended, it is intended for you to stretch yourself.

## Task 1 — Game Trees, Information Sets and Posterior Distributions (10%)

- Familiarize yourself with game trees.
- Suppose you are Player 1. Given an opponent **agent**, compute the distribution over states for a particular Player 1 information set.

**Start here →** open `tasks/task1.ipynb`.

## Task 2 — Treeplexes, Sequence Form, and Finding Best Responses in Single Round Games (20%)

- Find the best response of one player to another player's sequence form strategy.
- Calculate the saddle-point residual of a strategy profile.
- Understand when we are working with behavioral or sequence form strategies.

**Start here →** open `tasks/task2.ipynb`. You will edit `student_attempts/solver.py`.

## Task 3 - Compute the Nash equilibrium for Single Round Games (30%)

Compute an approximate Nash equilibrium strategy for tiny_hana using either:
- **Counterfactual Regret Minimization (CFR)** (recommended) or
- **Sequence-form LP** (linear program over the extensive-form game tree), or
- **Double Oracle** methods (incremenetal strategy generation), or
- Any other method, but justify them in the report.

**Start here →** open `tasks/task3.ipynb`. You will edit `student_attempts/solver.py`.

---

## Task 4 — Extensions (open-ended, 40%)

Explore some new topics. You are **not** required to do all of them.

- Solve the full multi-round game for tiny_hana. Each round's outcome determines the starting favor state for the next. This requires accounting for future-round value — a stochastic game where per-round strategies interact across rounds (possibly by solving a Markov Game)
- Scaling up to medium_hana (possibly hard!)
- Implementation speedups by exploiting the game structure (e.g., utilizing symmetry)
- Algorithmic speedups for "exact" methods (e.g., CFR+, MCCFR, linear (or more aggressive averaging)) 
- Approximate methods (e.g., subgame solving, machine-learning based algorithms, MARL, Deep CFR etc.)

Details are given in `tasks/task4.ipynb`, which also shows how to save a strategy as a policy table and play it in the tournament and the GUI.

Justifying and explaining what you have done for this task will have to be done in the **report**. For example, don't just claim that you have implemented Monte Carlo CFR, explain to me how it works and how it relates to your implementation.

It is alright if what you have tried does not work. However, you have to explain or give potential reasons for the failure and back it up with experimental results, e.g., runtime plots.

You are also allowed to explore other areas of game solving, but do check with me beforehand. 

---

*Tasks increase in difficulty and depend on the previous ones. It is recommended that you complete them in order.*
