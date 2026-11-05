import copy
from collections import defaultdict
from env_config import env, PLAYER_1, PLAYER_2
from engine.info_state import RoundInfoState
import numpy as np
from typing import Any, Dict

def interactive_game_tree_traversal():
    state = env.GameState()

    while not state.is_round_complete():
        print(state.pretty_print())

        if state.current_player == env.CHANCE_PLAYER:
            actions, probs = map(list, zip(*state.get_chance_actions_with_probs()))
            for action_id, action in enumerate(actions):
                print(f"[{action_id}] Chance Action: {action}, Probability: {probs[action_id]}")
        else:
            actions = list(state.get_legal_actions())
            for action_id, action in enumerate(actions):
                print(f"[{action_id}] Action for Player {state.current_player}: {action}")
        
        action = input(f"Enter action for Player {state.current_player} (or 'q' to quit): ")
        if action.lower() == 'q':
            break
        else:
            selected_action_id = int(action)
            state.execute_action(actions[selected_action_id])

    print("Final state *before* updating favors:")
    print(state.pretty_print())
    print("*"*50)

    print("Final state *after* updating favors:")
    state.update_favors()
    print(state.pretty_print())

    print(f"Winner: {state.check_winner()}")

def compute_number_of_leaves():
    """
    This function provides a sample of how one may want to traverse a game tree.

    It traverses the tree recursively (not efficient!) in a depth first fashion and counts the number of terminal states.

    It also illustrates the "undo_move" functionality, which tends to be faster than performing a deepcopy of 
    the state and performing search on that new state. Not all environments will allow for a simple implementation of
    undo_move.
    """
    state = env.GameState()

    def count_leaves(state):
        if state.is_round_complete():
            return 1
        total_leaves = 0
        if state.current_player == env.CHANCE_PLAYER:
            for action, _ in state.get_chance_actions_with_probs():
                if not state.execute_action(action):
                    raise RuntimeError(f"Failed to execute chance action during traversal: {action}")
                try:
                    total_leaves += count_leaves(state)
                finally:
                    state.undo_move()
        else:
            for action in state.get_legal_actions():
                if not state.execute_action(action):
                    raise RuntimeError(f"Failed to execute player action during traversal: {action}")
                try:
                    total_leaves += count_leaves(state)
                finally:
                    state.undo_move()
        return total_leaves

    total_leaves = count_leaves(state)
    print(f"Total number of leaves in the game tree: {total_leaves}")



def compute_agent_payoffs(initial_state: env.GameState,
                    p1_agent: env.Agent,
                    p2_agent: env.Agent) -> float:

    def traverse_tree(state: env.GameState, prob: float)->float:
        pay = 0.0
        if state.is_round_complete():
            return prob * state.terminal_payoff(player_id=PLAYER_1)
        
        if state.current_player == env.CHANCE_PLAYER:
            for action, action_prob in state.get_chance_actions_with_probs():
                if not state.execute_action(action):
                    raise RuntimeError(f"Failed to execute chance action during traversal: {action}")
                try:
                    pay += traverse_tree(state, prob * action_prob)
                finally:
                    state.undo_move()
        elif state.current_player == PLAYER_1:
            info_state = state.get_info_state(PLAYER_1)
            for action, action_prob in p1_agent.get_action_distribution(info_state).items():
                if action_prob == 0.0:
                    continue
                if not state.execute_action(action):
                    raise RuntimeError(f"Player 1 strategy returned illegal action: {action}")
                try:
                    pay += traverse_tree(state, prob * action_prob)
                finally:
                    state.undo_move()
        elif state.current_player == PLAYER_2:
            info_state = state.get_info_state(PLAYER_2)
            for action, action_prob in p2_agent.get_action_distribution(info_state).items():
                if action_prob == 0.0:
                    continue
                if not state.execute_action(action):
                    raise RuntimeError(f"Player 2 strategy returned illegal action: {action}")
                try:
                    pay += traverse_tree(state, prob * action_prob)
                finally:
                    state.undo_move()
        else:
            assert False, "Invalid player"

        return pay

    start_state = copy.deepcopy(initial_state)
    total_payoff = traverse_tree(start_state, 1.0)

    return total_payoff

def compute_posterior_given_opponent_agent(initial_state: env.GameState, 
                      target_info_state: RoundInfoState,
                      opponent_agent: env.Agent) -> Dict[Any, float]:
    
    """
    Given an initial game state and a target information state, as well as the opponent's strategy,
    compute the posterior distribution over the possible game states that are consistent with the target information state.

    Args:
        initial_state: GameState object representing the initial state of the game
        target_info_state: RoundInfoState object representing the information state for which we want to compute the posterior
        opponent_agent: Agent object representing the opponent's strategy
    
    Returns:
        A dictionary (or defaultdict) that maps `state key`s to the probability of that state being the true state given 
        the target information state and opponent strategy. Leave out states with probability 0, and return an empty
        dictionary if the opponent's strategy never reaches the target information state.

    Hint: 
        - For any state, use state.to_key() to obtain its `state key`.
        - Refer to `compute_number_of_leaves` and `compute_agent_payoffs` for examples of how to traverse the game tree. 
    """

    target_player = target_info_state.player_id
    opponent_player = 1 - target_player

    # Your code here: traverse the game tree and compute the posterior distribution over game states consistent with target_info_state
    raise NotImplementedError("Posterior computation not implemented yet")

def generate_random_infoset(prob_terminate_anywhere = 0.1, seed=12345):
    import random
    rng = random.Random(seed)    
  
    while True:
        state = env.GameState()
        while not state.is_round_complete():
            if state.current_player == env.CHANCE_PLAYER:
                actions, probs = zip(*state.get_chance_actions_with_probs())
                action = rng.choices(actions, weights=probs)[0]
                if not state.execute_action(action):
                    raise RuntimeError(f"Failed to execute chance action during traversal: {action}")
            else:
                # Decide if we should terminate and return this state
                terminate = rng.choices([True, False], weights=[prob_terminate_anywhere, 1-prob_terminate_anywhere], k=1)[0]
                if terminate: 
                    return state.get_info_state(state.current_player)

                actions = list(state.get_legal_actions())
                action = rng.choices(actions)[0]
                if not state.execute_action(action):
                    raise RuntimeError(f"Failed to execute player action during traversal: {action}")

def generate_random_infosets(num, seed=12345, prob=0.1):
    infosets = []
    for k in range(num):
        infosets.append(generate_random_infoset(prob, seed+k))
    return infosets
