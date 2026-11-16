import tiny_hana as env  # Change this line to switch environments (e.g. medium_hana)

from engine.constants import PLAYER_1, PLAYER_2


def make_round_start_state(starting_player: int, favors: dict = None) -> env.GameState:
    """Return a GameState at the start of a round, before cards are dealt.

    The returned state is at phase="deal_initial": current_player is CHANCE_PLAYER,
    ready to receive CHANCE_DEAL actions (or to pass directly to TreeplexGame).

    Args:
        starting_player: 0 or 1 — the player who acts first this round.
        favors: optional dict mapping each Suit to an int:
                  -1  -> PLAYER_1 controls this geisha
                   0  -> neutral (default for all suits if omitted)
                  +1  -> PLAYER_2 controls this geisha
                Keys not present keep their default (0 = neutral).
    """
    state = env.GameState()
    if favors is not None:
        state.favors = {**state.favors, **favors}   # assign: works with either engine
    state.execute_action(env.Action(env.ActionType.CHANCE_START, [], choice=starting_player))
    return state
