import numpy as np
from env_config import PLAYER_1, PLAYER_2
from treeplex_representation import TreeplexGame
from treeplex_representation import convert_sequence_to_behavioral_form_strategy, convert_behavioral_to_sequence_form_strategy

class Solver(object):
    def __init__(self, treeplex_game: TreeplexGame):
        self.game = treeplex_game

    def compute_payoffs_from_seq_form_strategies(self,
                                                 seq_form_strategy_p1: np.ndarray,
                                                 seq_form_strategy_p2: np.ndarray):
        total_payoff_p1 = 0.0
        for (seq_p1, seq_p2), payoff in self.game.seq_pair_to_leaves.items():
            total_payoff_p1 += seq_form_strategy_p1[seq_p1] * seq_form_strategy_p2[seq_p2] * payoff
            
        return total_payoff_p1

    def compute_reward_vector(self, 
                              player_id: int, 
                              opponent_seq_form_strategy: np.ndarray)->np.ndarray:
        """
        Computes a reward vector for the given player against the opponent's strategy. 
        The reward vector is indexed by the player's sequences and gives the expected payoff for each sequence 
        when the opponent plays according to the given strategy. Since we are always returning *rewards* for PLAYER_1, 
        so if player_id == 1 (i.e., PLAYER_2), we do a sign inversion to convert from costs to rewards since player 2 is assumed to be minimizing.

        The opponent strategy is **required** to be in sequence form.
        
        Args:
            player_id: PLAYER_1 or PLAYER_2, indicating which player's reward vector to compute.
            opponent_seq_form_strategy: A numpy array representing the opponent's strategy in sequence form. 
                The size of this array should match the number of sequences for the opponent's treeplex.
        
        Returns:
            A numpy array representing the reward vector for the given player against the opponent's strategy, 
            indexed by player_id's sequences. The size of this array will match the number of sequences for the player's treeplex.
        """

        if player_id == PLAYER_1:
            assert opponent_seq_form_strategy.size == self.game.treeplex_p2.num_sequences, "Opponent strategy size does not match their treeplex size"
            reward_vec = np.zeros(self.game.treeplex_p1.num_sequences)
            for (seq_p1, seq_p2), payoff in self.game.seq_pair_to_leaves.items():
                reward_vec[seq_p1] += opponent_seq_form_strategy[seq_p2] * payoff
            return reward_vec
        elif player_id == PLAYER_2:
            assert opponent_seq_form_strategy.size == self.game.treeplex_p1.num_sequences, "Opponent strategy size does not match their treeplex size"
            reward_vec = np.zeros(self.game.treeplex_p2.num_sequences)
            for (seq_p1, seq_p2), payoff in self.game.seq_pair_to_leaves.items():
                reward_vec[seq_p2] -= opponent_seq_form_strategy[seq_p1] * payoff # Player 2 is assumed to be minimizing, so we do a sign inversion.
            return reward_vec

    def compute_best_response(
            self,
            player_id: int,
            opponent_seq_form_strategy: np.ndarray,
            convert_to_seq_form: bool = False) -> np.ndarray:
        
        """
        Args:
            player_id: PLAYER_1 or PLAYER_2, indicating which player is the best-responding one
            opponent_seq_form_strategy: A numpy array representing the opponent's strategy in sequence form.
            convert_to_seq_form: If True, convert the resulting best response from behavioral form to sequence form before returning. 
            If False, return the best response in behavioral form. By default, return behavioral strategies.
        
        Returns:
            numpy array representing the best response strategy for the given player against the opponent's strategy. 
            The returned strategy will be in behavioral form if convert_to_seq_form=False, and in sequence form if convert_to_seq_form=True.
        """

        vec = self.compute_reward_vector(player_id, opponent_seq_form_strategy)
        
        # Your code here: compute the best response strategy for player_id against the opponent's strategy given the reward vector.

        raise NotImplementedError("compute_best_response not implemented yet")

    def saddle_point_residual(self, sol_p1: np.ndarray, sol_p2: np.ndarray):
        """
        Computes the saddle point residual (exploitability) of the given strategy profile. Note that the input strategies are expected to be in sequence form.

        Args:
            sol_p1: A numpy array representing player 1's strategy in sequence form.
            sol_p2: A numpy array representing player 2's strategy in sequence form.
        
        Returns:
            A float/numpy scalar representing the saddle point residual (exploitability) of the strategy profile (sol_p1, sol_p2).
        """

        # Your code here

        raise NotImplementedError("saddle_point_residual not implemented yet")


class CFRSolver(Solver):
    def __init__(self, treeplex_game: TreeplexGame):
        super().__init__(treeplex_game)

    def solve(self, max_iter=1000, spr_threshold=1e-2, verbose: bool = False, reset: bool = False):
        """
        Run CFR for up to max_iter iterations, stopping early when SPR <= spr_threshold.

        Optionally, keep the regrets between calls so that a second solve() continues where the
        first stopped (useful when the game changes slowly, e.g. in value iteration for Task 4);
        reset=True would then start again from zero regrets.

        Returns:
            (sol_seq_form_p1, sol_seq_form_p2): **time-average** sequence-form strategies.

        After returning, sets self.spr, self.val, self.sol_seq_form_p1, self.sol_seq_form_p2.
        """
        raise NotImplementedError("CFRSolver.solve() not implemented yet")
