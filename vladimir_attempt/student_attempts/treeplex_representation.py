import numpy as np
from collections import defaultdict
from env_config import env, PLAYER_1, PLAYER_2
from engine import require_enumerable
from typing import Dict
import copy

Agent = env.Agent

class Treeplex(object):
    def __init__(self): 

        self.num_sequences = 1
        self.info_state_start_seq = []
        self.info_state_end_seq = []
        self.info_state_parent_seq = []

    def add_infoset(self, parent_seq, num_actions):
        self.info_state_start_seq.append(self.num_sequences)
        self.num_sequences += num_actions
        self.info_state_end_seq.append(self.num_sequences)

        self.info_state_parent_seq.append(parent_seq)

    def get_num_infosets(self):
        return len(self.info_state_end_seq)



class TreeplexGame(object):
    def __init__(self, initial_state: env.GameState = None):
        self.extract_treeplex(initial_state)

    def extract_treeplex(self, initial_state: env.GameState = None):
        state = initial_state if initial_state is not None else env.GameState()
        require_enumerable(getattr(state, "config", None), "treeplex extraction")

        treeplex_p1, treeplex_p2 = Treeplex(), Treeplex()

        # For reconstruction
        self.info_states_id_to_key_p1 = []
        self.info_states_id_to_key_p2 = []
        
        self.info_states_key_to_id_p1 = dict()
        self.info_states_key_to_id_p2 = dict()

        # For payoffs
        self.seq_pair_to_leaves = defaultdict(float)
        self.seq_to_leaves_p1 = defaultdict(set)
        self.seq_to_leaves_p2 = defaultdict(set)

        def traverse_tree(state: env.GameState, seq_id_p1: int, seq_id_p2: int, cum_prob: float) -> None:
            nonlocal self, treeplex_p1, treeplex_p2
            if state.is_round_complete():
                #Record into leaf structure
                seq_pair = seq_id_p1, seq_id_p2
                self.seq_pair_to_leaves[seq_pair] += cum_prob * state.terminal_payoff(player_id=PLAYER_1)
                self.seq_to_leaves_p1[seq_id_p1].add(seq_id_p2)
                self.seq_to_leaves_p2[seq_id_p2].add(seq_id_p1)
                return 

            if state.current_player == env.CHANCE_PLAYER:
                for action, action_prob in state.get_chance_actions_with_probs():
                    state.execute_action(action)
                    try:
                        traverse_tree(state, seq_id_p1, seq_id_p2, cum_prob * action_prob)
                    finally:
                        state.undo_move()
            elif state.current_player == PLAYER_1:
                info_state = state.get_info_state(PLAYER_1)
                if info_state not in self.info_states_key_to_id_p1:
                    # Need to create new infostate
                    self.info_states_key_to_id_p1[info_state] = len(self.info_states_key_to_id_p1)
                    self.info_states_id_to_key_p1.append(info_state)

                    treeplex_p1.add_infoset(seq_id_p1, len(list(state.get_legal_actions())))

                info_state_id = self.info_states_key_to_id_p1[info_state]

                # Still need to recurse to future parts of the tree
                # even if this infoset was encountered before (might get new info states downstream)
                for action_index, action in enumerate(state.get_legal_actions()):
                    new_seq_id_p1 = treeplex_p1.info_state_start_seq[info_state_id] + action_index
                    state.execute_action(action)
                    try:
                        traverse_tree(state, new_seq_id_p1, seq_id_p2, cum_prob)
                    finally:
                        state.undo_move()
            elif state.current_player == PLAYER_2:
                info_state = state.get_info_state(PLAYER_2)
                if info_state not in self.info_states_key_to_id_p2:
                    # Need to create new infostate
                    self.info_states_key_to_id_p2[info_state] = len(self.info_states_key_to_id_p2)
                    self.info_states_id_to_key_p2.append(info_state)

                    treeplex_p2.add_infoset(seq_id_p2, len(list(state.get_legal_actions())))

                info_state_id = self.info_states_key_to_id_p2[info_state]

                # Still need to recurse to future parts of the tree
                # even if this infoset was encountered before (might get new info states downstream)
                for action_index, action in enumerate(state.get_legal_actions()):
                    new_seq_id_p2 = treeplex_p2.info_state_start_seq[info_state_id] + action_index
                    state.execute_action(action)
                    try:
                        traverse_tree(state, seq_id_p1, new_seq_id_p2, cum_prob)
                    finally:
                        state.undo_move()
            else:
                assert False, "Invalid player"

        traverse_tree(state, 0, 0, 1.0)    

        self.treeplex_p1 = treeplex_p1
        self.treeplex_p2 = treeplex_p2


    def agent_to_behavioral_strategy(self, agent: Agent, player_id: int) -> np.ndarray:
        if player_id == PLAYER_1:
            treeplex = self.treeplex_p1
            info_state_id_to_key = self.info_states_id_to_key_p1
        elif player_id == PLAYER_2:
            treeplex = self.treeplex_p2
            info_state_id_to_key = self.info_states_id_to_key_p2
        else:
            assert False, 'Invalid player id'

        behavioral_vec = np.zeros(treeplex.num_sequences)
        for infoset_id in range(treeplex.get_num_infosets()):
            info_state_key = info_state_id_to_key[infoset_id]
            dist = agent.get_action_distribution(info_state_key)
            
            for action_id, action in enumerate(info_state_key.get_legal_actions()):
                if action not in dist: continue
                behavioral_vec[treeplex.info_state_start_seq[infoset_id] + action_id] = dist[action]
        return behavioral_vec
            
        
    def behavioral_strategy_to_agent(self, strategy: np.ndarray, player_id: int) -> Agent:        
        class ReturnAgent(Agent):
            def __init__(agent_self, strategy: np.ndarray, player_id: int):
                agent_self.strategy = strategy
                agent_self.player_id = player_id

            def get_action_distribution(agent_self, info_state: env.RoundInfoState) -> Dict[env.Action, float]:
                if agent_self.player_id == PLAYER_1:
                    treeplex = self.treeplex_p1
                    info_state_key_to_id = self.info_states_key_to_id_p1
                elif agent_self.player_id == PLAYER_2:
                    treeplex = self.treeplex_p2
                    info_state_key_to_id = self.info_states_key_to_id_p2
                else:
                    assert False, 'Invalid player id'

                if info_state not in info_state_key_to_id:
                    raise ValueError("Info state not found in treeplex")

                infoset_id = info_state_key_to_id[info_state]
                start_seq_id, end_seq_id = treeplex.info_state_start_seq[infoset_id], treeplex.info_state_end_seq[infoset_id]

                action_dist = dict()
                for action_index, action in enumerate(info_state.get_legal_actions()):
                    if agent_self.strategy[start_seq_id + action_index] > 0:
                        action_dist[action] = agent_self.strategy[start_seq_id + action_index]
                return action_dist

        return ReturnAgent(strategy, player_id)

    def save_agent(self, agent: Agent, player_id: int, path: str) -> None:
        strategy = self.agent_to_behavioral_strategy(agent, player_id)
        np.savez(path, strategy=strategy, player_id=np.array([player_id]))

    def load_agent(self, path: str) -> Agent:
        data = np.load(path)
        strategy = data['strategy']
        player_id = int(data['player_id'][0])
        if player_id == PLAYER_1:
            expected_size = self.treeplex_p1.num_sequences
        elif player_id == PLAYER_2:
            expected_size = self.treeplex_p2.num_sequences
        else:
            raise ValueError(f"Invalid player_id {player_id} in saved file.")
        if strategy.size != expected_size:
            raise ValueError(
                f"Strategy size {strategy.size} does not match treeplex size {expected_size} "
                f"for player {player_id}. File may be from a different game variant."
            )
        return self.behavioral_strategy_to_agent(strategy, player_id)


def convert_behavioral_to_sequence_form_strategy(strat: np.ndarray, 
                                                 treeplex: Treeplex, 
                                                 in_place: bool=True,
                                                 ):
    if not in_place:
        new_strat = np.copy(strat)
        convert_behavioral_to_sequence_form_strategy(new_strat, treeplex, in_place=True)
        return new_strat
    else:
        strat[0] = 1.0
        for infoset_id in range(treeplex.get_num_infosets()):
            start_seq_id, end_seq_id = treeplex.info_state_start_seq[infoset_id], treeplex.info_state_end_seq[infoset_id]
            parent_seq_id = treeplex.info_state_parent_seq[infoset_id]
            assert np.all(strat[start_seq_id: end_seq_id] >= 0.0)           
            assert np.isclose(np.sum(strat[start_seq_id:end_seq_id]), 1.0), f"Behavioral strategy probabilities for infoset {infoset_id} do not sum to 1. Got sum {np.sum(strat[start_seq_id:end_seq_id])}"

            strat[start_seq_id:end_seq_id] *= strat[parent_seq_id]

def convert_sequence_to_behavioral_form_strategy(strat:np.ndarray, 
                                                 treeplex: Treeplex, 
                                                 in_place: bool=True, 
                                                 eps=1e-6):
    if not in_place:
        new_strat = np.copy(strat)
        convert_sequence_to_behavioral_form_strategy(new_strat, treeplex, in_place=True, eps=eps)
        return new_strat
    else:
        strat[0] = 1.0
        for infoset_id in reversed(range(treeplex.get_num_infosets())):
            start_seq_id, end_seq_id = treeplex.info_state_start_seq[infoset_id], treeplex.info_state_end_seq[infoset_id]
            parent_seq_id = treeplex.info_state_parent_seq[infoset_id]
            assert np.isclose(np.sum(strat[start_seq_id:end_seq_id]), strat[parent_seq_id])
            
            if strat[parent_seq_id] >= eps:
                strat[start_seq_id:end_seq_id] /= strat[parent_seq_id]
            else:
                strat[start_seq_id:end_seq_id] = 1.0 / (end_seq_id-start_seq_id)
