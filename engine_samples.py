"""
engine_samples.py — Annotated examples of the Hanamikoji engine API.

Covers four combinations:
  - tiny_hana  (3 suits, RESERVE + SPLIT)     with Python backend
  - medium_hana (5 suits, RESERVE + DISCARD + SPLIT) with Python backend
  - tiny_hana  with C++ backend
  - medium_hana with C++ backend

Two usage levels are shown for each:
  HIGH-LEVEL   Game.play_game()  — for running matches
  LOW-LEVEL    GameState loop    — for solvers / MCTS / tree search

Run in Python mode (default):
    python engine_samples.py

Run in C++ mode:
    HANA_ENGINE=cpp python engine_samples.py

The C++ backend activates when HANA_ENGINE=cpp is set *before* the first
import of `engine`.  In C++ mode all the same high-level code works
unchanged; we additionally show the solver extras (undo_move, to_key,
get_chance_actions_with_probs), which the Python engine has as well.
"""

import os
import random

_CPP = os.environ.get("HANA_ENGINE") == "cpp"

print("=" * 60)
print(f"Backend: {'C++ (HANA_ENGINE=cpp)' if _CPP else 'Python'}")
print("=" * 60)


# ---------------------------------------------------------------------------
# Custom agent shared by both variants
#
# ReserveHighestRankAgent: if RESERVE is available and not yet used, reserve
# the highest-rank card in hand; otherwise fall back to a uniform random choice.
#
# This is a genuine Agent subclass — it works identically regardless of which
# backend is active, because it only uses the RoundInfoState API.
# ---------------------------------------------------------------------------

from engine import Agent, Action, CHANCE_PLAYER  # noqa: E402


class ReserveHighestRankAgent(Agent):
    """Prefer RESERVE on the best card; otherwise pick uniformly at random."""

    def get_action_distribution(self, info_state):
        legal = list(info_state.get_legal_actions())

        # Try to find a RESERVE action for the highest-rank card in hand.
        reserve_actions = [
            a for a in legal
            if a.action_type.name == "RESERVE"
        ]
        if reserve_actions:
            # Pick the RESERVE action whose card has the highest suit value.
            # suit.value is an int (0 = rank-1 suit, 4 = rank-4 suit for medium).
            best = max(reserve_actions, key=lambda a: a.cards[0].suit.value)
            return {best: 1.0}

        # No RESERVE available — uniform over all legal actions.
        p = 1.0 / len(legal)
        return {a: p for a in legal}


# ---------------------------------------------------------------------------
# Helper: run one low-level game loop manually.
#
# This is the pattern a solver (MCTS, CFR) would use:
#   1. Drive chance nodes with sample_chance_action / a random choice.
#   2. At player nodes, ask the agent for a distribution and sample.
#   3. At ROUND_COMPLETE, call update_favors(), check_winner(), begin_next_round().
#
# In C++ mode we additionally call undo_move() to show that the API is the
# same as Python's undo_move().
# ---------------------------------------------------------------------------

def run_low_level_game(GameState, config, agents, rng, label):
    """Drive a full game through the GameState API and return the winner."""
    print(f"\n  [{label}] low-level game loop")

    gs = GameState(config)

    # C++ mode only: snapshot the initial key so we can demonstrate undo.
    if _CPP:
        initial_key = gs.to_key()

    rounds_played = 0
    while True:
        # ── Chance node ──────────────────────────────────────────────────
        cp = gs.current_player
        if cp == CHANCE_PLAYER:
            if _CPP:
                # C++ exposes a fast seeded sampler and exact probabilities.
                # get_chance_actions_with_probs() returns [(action, prob), ...]
                # which is what an expectimax solver would enumerate.
                from engine import Rng
                cpp_rng = Rng(rng.randint(0, 2**63))
                action = gs.sample_chance_action(cpp_rng)
            else:
                # Python: pick uniformly from the legal chance actions.
                actions = list(gs.get_legal_actions())
                action = rng.choice(actions)

            gs.execute_action(action)
            continue

        # ── Round complete ────────────────────────────────────────────────
        if cp is None:
            gs.update_favors()
            winner = gs.check_winner()
            # Python returns None for "no winner yet"; C++ returns -2 (NO_PLAYER).
            # Normalise both to None so the rest of the loop is backend-agnostic.
            if winner == -2:
                winner = None
            if winner is not None:
                print(f"  [{label}] winner after round {gs.round_count}: Player {winner}")
                return winner
            rounds_played += 1
            if rounds_played > 10:
                raise RuntimeError("Game exceeded 10 rounds — possible infinite loop")
            # C++ exposes begin_next_round(); Python uses the internal _begin_round().
            if _CPP:
                gs.begin_next_round()
            else:
                gs._begin_round()
            continue

        # ── Player node ───────────────────────────────────────────────────
        info = gs.get_info_state(cp)
        distribution = agents[cp].get_action_distribution(info)

        # Sample from the distribution.
        actions, weights = zip(*distribution.items())
        (action,) = rng.choices(actions, weights=weights, k=1)

        # C++ only: demonstrate undo — take the action and immediately undo
        # it on the very first player decision of the game, then re-execute.
        if _CPP and rounds_played == 0 and gs.turn_count == 0:
            key_before = gs.to_key()
            gs.execute_action(action)
            gs.undo_move()
            assert gs.to_key() == key_before, "undo_move did not restore state!"
            print(f"  [{label}] undo_move verified: key restored after execute+undo")

        gs.execute_action(action)


# ===========================================================================
# ███  SECTION 1 — tiny_hana, Python interface
# ===========================================================================

print("\n\n" + "=" * 60)
print("SECTION 1 — tiny_hana, Python backend")
print("=" * 60)

# tiny_hana exports its own GameState/Game/Suit subclasses pre-configured
# with GAME_CONFIG (3 suits: GEISHA_1=2 cards, GEISHA_2=2, GEISHA_3=3;
# available actions: RESERVE + SPLIT; win: 2+ geishas).
from tiny_hana.tiny_hana import (     # noqa: E402
    Game as TinyGame,
    GameState as TinyGameState,
    GAME_CONFIG as TINY_CONFIG,
    RandomAgent as TinyRandomAgent,
    Suit as TinySuit,
)

# ── 1a. High-level: play a complete game via Game.play_game() ────────────────
# Game() handles chance nodes, calls agents at decision points, logs output.
# NOTE: verbose=True produces Python stdout logging in Python mode.
#       In C++ mode the C++ game runner runs the loop internally and the
#       verbose flag is currently a no-op (placeholder for future logging).
print("\n--- 1a. High-level game (verbose in Python mode) ---")
a0 = TinyRandomAgent()
a1 = ReserveHighestRankAgent()
game = TinyGame(a0, a1, verbose=not _CPP, seed=0)
winner = game.play_game()
print(f"Winner: Player {winner}")

# ── 1b. Low-level: manual GameState loop (what a solver uses) ───────────────
# We manage the game loop ourselves, calling execute_action() directly.
print("\n--- 1b. Low-level GameState loop ---")
rng_1b = random.Random(1)
winner_1b = run_low_level_game(
    TinyGameState, TINY_CONFIG,
    [ReserveHighestRankAgent(), TinyRandomAgent()],
    rng_1b, "tiny/py"
)

# ── 1c. Inspect info state — what the agent actually sees ───────────────────
print("\n--- 1c. Info state inspection ---")
gs_inspect = TinyGameState()
rng_inspect = random.Random(42)

# Advance past chance nodes to the first player decision.
while gs_inspect.current_player == CHANCE_PLAYER:
    actions = list(gs_inspect.get_legal_actions())
    gs_inspect.execute_action(rng_inspect.choice(actions))

pid = gs_inspect.current_player
info = gs_inspect.get_info_state(pid)

hand, used = info.get_current_hand_and_actions()
favors = info.get_favors()

print(f"  Player {pid}'s perspective:")
print(f"  Hand : {[c.suit.name for c in hand]}")
print(f"  Used actions : {[at.name for at in used]}")
print(f"  Favors : { {s.name: v for s, v in favors.items()} }")
print(f"  Legal actions: {[str(a) for a in info.get_legal_actions()]}")
# The full info state string is the key used in policy-table artifacts.
print(f"  Info state key (first 80 chars): {str(info)[:80]}")


# ===========================================================================
# ███  SECTION 2 — medium_hana, Python interface
# ===========================================================================

print("\n\n" + "=" * 60)
print("SECTION 2 — medium_hana, Python backend")
print("=" * 60)

# medium_hana adds a DISCARD action and a second win condition (6+ favor pts).
# Everything else is the same API — the variant is captured in GAME_CONFIG.
from medium_hana.medium_hana import (  # noqa: E402
    Game as MediumGame,
    GameState as MediumGameState,
    GAME_CONFIG as MEDIUM_CONFIG,
    RandomAgent as MediumRandomAgent,
    Suit as MediumSuit,
)

# ── 2a. High-level ──────────────────────────────────────────────────────────
print("\n--- 2a. High-level game (verbose in Python mode) ---")
game_med = MediumGame(ReserveHighestRankAgent(), MediumRandomAgent(),
                      verbose=not _CPP, seed=5)
winner_med = game_med.play_game()
print(f"Winner: Player {winner_med}")

# ── 2b. Low-level loop — medium_hana has DISCARD so the legal-action set
#        is larger (up to 3 action types per player per round).  Everything
#        else in the loop is identical to tiny_hana.
print("\n--- 2b. Low-level GameState loop ---")
rng_2b = random.Random(2)
winner_2b = run_low_level_game(
    MediumGameState, MEDIUM_CONFIG,
    [ReserveHighestRankAgent(), MediumRandomAgent()],
    rng_2b, "medium/py"
)

# ── 2c. Show DISCARD-specific state after the action ────────────────────────
print("\n--- 2c. DISCARD action in medium_hana ---")
gs_dis = MediumGameState()
rng_dis = random.Random(7)

while gs_dis.current_player == CHANCE_PLAYER:
    actions = list(gs_dis.get_legal_actions())
    gs_dis.execute_action(rng_dis.choice(actions))

pid_dis = gs_dis.current_player
info_dis = gs_dis.get_info_state(pid_dis)
legal_dis = list(info_dis.get_legal_actions())

# Find a DISCARD action to execute.
discard_action = next(
    (a for a in legal_dis if a.action_type.name == "DISCARD"), None
)
if discard_action:
    gs_dis.execute_action(discard_action)
    player_state = gs_dis.players[pid_dis]
    print(f"  Player {pid_dis} played DISCARD:")
    print(f"  Revealed card : {player_state.discarded_revealed[0].suit.name}")
    print(f"  Hidden  card  : (hidden — opponent sees only revealed)")
    print(f"  Hand after    : {[c.suit.name for c in player_state.hand]}")
else:
    print("  (No DISCARD action available at this node — try a different seed)")


# ===========================================================================
# ███  SECTION 3 — tiny_hana, C++ backend
# ===========================================================================

print("\n\n" + "=" * 60)
print("SECTION 3 — tiny_hana, C++ backend")
if not _CPP:
    print("  (skipped — run with HANA_ENGINE=cpp to enable)")
    print("=" * 60)
else:
    print("=" * 60)

    import hana_engine_cpp as _cpp  # noqa: E402 — C++ extension

    # When HANA_ENGINE=cpp, engine.* returns C++ types.
    # TinyGame and TinyGameState (imported above) already use these because
    # they import from `engine`, which dispatched to the C++ backend.
    # Confirm this:
    from engine import GameState as EngineGS  # noqa: E402
    assert issubclass(TinyGameState, _cpp.GameState), \
        "Expected TinyGameState to be a C++ subclass"
    print(f"\n  TinyGameState base class: {TinyGameState.__bases__}")

    # ── 3a. High-level — identical call, C++ types underneath ────────────────
    print("\n--- 3a. High-level game (C++ verbose is a no-op) ---")
    game_cpp_tiny = TinyGame(TinyRandomAgent(), ReserveHighestRankAgent(),
                             verbose=False, seed=10)
    w3a = game_cpp_tiny.play_game()
    print(f"Winner: Player {w3a}")

    # ── 3b. Low-level with C++-only solver extras ─────────────────────────────
    # Extra features available only in C++ mode:
    #
    #   gs.to_key()                        — value snapshot, usable as dict key
    #   gs.undo_move()                     — exact state reversal (O(1))
    #   gs.get_chance_actions_with_probs() — [(action, prob)] for expectimax
    #   gs.sample_chance_action(Rng(seed)) — fast seeded chance sampling
    #   gs.preview_round_resolution()      — winner/favors without mutating state
    #
    print("\n--- 3b. Low-level loop with solver extras ---")
    from engine import Rng  # noqa: E402

    gs3 = TinyGameState()
    cpp_rng = Rng(42)

    # Enumerate chance actions with exact probabilities at CHANCE_START.
    pairs = gs3.get_chance_actions_with_probs()
    print(f"  CHANCE_START actions with probs: "
          f"{[(a.action_type.name, f'{p:.4f}') for a, p in pairs]}")

    # Drive the game forward.
    rng_3b = random.Random(10)
    w3b = run_low_level_game(
        TinyGameState, TINY_CONFIG,
        [TinyRandomAgent(), ReserveHighestRankAgent()],
        rng_3b, "tiny/cpp"
    )

    # ── 3c. Transposition table demo ─────────────────────────────────────────
    print("\n--- 3c. to_key() as transposition table key ---")
    gs_key = TinyGameState()
    # Build a tiny table: map state key → a float (e.g. a cached value).
    table = {}
    cpp_rng2 = Rng(99)
    while gs_key.current_player == CHANCE_PLAYER:
        gs_key.execute_action(gs_key.sample_chance_action(cpp_rng2))

    key0 = gs_key.to_key()
    table[key0] = 0.75   # pretend this is a cached minimax value

    # undo_move works through chance nodes too, but only player/chance nodes
    # after CHOOSE_START are undoable.  Instead demonstrate lookup:
    assert table[gs_key.to_key()] == 0.75
    print(f"  Cached value retrieved from table: {table[gs_key.to_key()]}")

    legal3c = list(gs_key.get_legal_actions())
    for act in legal3c:
        gs_key.execute_action(act)
        table[gs_key.to_key()] = 0.0   # stub value
        gs_key.undo_move()
    print(f"  After probing {len(legal3c)} children and undoing: "
          f"state restored → {gs_key.to_key() == key0}")
    print(f"  Table size: {len(table)} entries")


# ===========================================================================
# ███  SECTION 4 — medium_hana, C++ backend
# ===========================================================================

print("\n\n" + "=" * 60)
print("SECTION 4 — medium_hana, C++ backend")
if not _CPP:
    print("  (skipped — run with HANA_ENGINE=cpp to enable)")
    print("=" * 60)
else:
    print("=" * 60)

    # ── 4a. High-level ────────────────────────────────────────────────────────
    print("\n--- 4a. High-level game ---")
    game_cpp_med = MediumGame(ReserveHighestRankAgent(), MediumRandomAgent(),
                              verbose=False, seed=20)
    w4a = game_cpp_med.play_game()
    print(f"Winner: Player {w4a}")

    # ── 4b. Direct _cpp API — bypass the engine shim entirely ─────────────────
    # You can also construct a C++ GameState directly using RuntimeConfig.
    # This is what the C++ solver would do without any Python wrapper overhead.
    print("\n--- 4b. Direct _cpp API (no Python shim) ---")

    cfg = _cpp.RuntimeConfig.medium_hana()
    print(f"  RuntimeConfig: {cfg.num_suits} suits, "
          f"win_favor_points={cfg.win_favor_points}, "
          f"win_geisha_count={cfg.win_geisha_count}")

    gs4 = _cpp.GameState(cfg)
    cpp_rng4 = Rng(55)

    # Advance through chance nodes.
    while gs4.current_player == CHANCE_PLAYER:
        gs4.execute_action(gs4.sample_chance_action(cpp_rng4))

    pid4 = gs4.current_player
    legal4 = list(gs4.get_legal_actions())
    action_names = sorted({a.action_type.name for a in legal4})
    print(f"  Player {pid4}'s available action types: {action_names}")
    print(f"  Total legal actions: {len(legal4)}")

    # Show info state via get_info_state (C++ RoundInfoState).
    info4 = gs4.get_info_state(pid4)
    hand4, used4 = info4.get_current_hand_and_actions()
    favors4 = info4.get_favors()
    print(f"  Hand  : {[c.suit.name for c in hand4]}")
    print(f"  Favors: { {s.name: v for s, v in favors4.items()} }")

    # Execute a random action, then preview_round_resolution after completion.
    rng_4b = random.Random(55)
    run_low_level_game(
        MediumGameState, MEDIUM_CONFIG,
        [ReserveHighestRankAgent(), MediumRandomAgent()],
        rng_4b, "medium/cpp"
    )

    # ── 4c. preview_round_resolution — inspect outcome without mutating ────────
    print("\n--- 4c. preview_round_resolution ---")
    gs4c = _cpp.GameState(cfg)
    cpp_rng4c = Rng(77)

    while not gs4c.is_round_complete():
        if gs4c.current_player == CHANCE_PLAYER:
            gs4c.execute_action(gs4c.sample_chance_action(cpp_rng4c))
        else:
            legal4c = list(gs4c.get_legal_actions())
            gs4c.execute_action(rng_4b.choice(legal4c))

    res = gs4c.preview_round_resolution()
    print(f"  Round outcome (without calling update_favors):")
    print(f"  Final favors : { {s.name: v for s, v in res.final_favors.items()} }")
    if res.winner is None:
        print(f"  Winner       : None (no threshold met this round)")
    else:
        # Determine whether it was a points-win or geisha-count-win.
        suit_ranks = [1, 1, 2, 3, 4]
        pts = sum(suit_ranks[s.value] for s, v in res.final_favors.items() if v == (-1 if res.winner == 0 else 1))
        print(f"  Winner       : Player {res.winner} "
              f"({'points: ' + str(pts) + ' ≥ 6' if pts >= 6 else 'geisha count ≥ 3'})")
    print(f"  State is still un-mutated: phase = {gs4c.phase}")


print("\n\nAll samples complete.")
