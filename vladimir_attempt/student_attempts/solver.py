"""Best responses and full-tree counterfactual regret minimization.

Only NumPy and the supplied game representation are needed at runtime.  The
optional CFR+ and averaging choices support the experiments in Task 4.
"""

import time
import numpy as np
from env_config import PLAYER_1, PLAYER_2
from treeplex_representation import TreeplexGame
from treeplex_representation import (
    convert_sequence_to_behavioral_form_strategy,
    convert_behavioral_to_sequence_form_strategy,
)


class _TreeplexOperations:
    """Batch independent infosets at equal own-action depth.

    Each sequence has a single parent infoset.  Processing a whole depth at a
    time is therefore equivalent to the supplied forward/backward traversals.
    ``np.add.at`` matters: several information sets can have the same parent.
    """

    def __init__(self, treeplex):
        self.treeplex = treeplex
        self.starts = np.asarray(treeplex.info_state_start_seq, dtype=np.intp)
        self.ends = np.asarray(treeplex.info_state_end_seq, dtype=np.intp)
        self.parents = np.asarray(treeplex.info_state_parent_seq, dtype=np.intp)
        self.counts = self.ends - self.starts
        self.owner = np.repeat(np.arange(len(self.starts)), self.counts)
        depths = np.zeros(treeplex.num_sequences, dtype=np.intp)
        info_depths = np.empty(len(self.starts), dtype=np.intp)
        for i, (start, end, parent) in enumerate(zip(self.starts, self.ends, self.parents)):
            info_depths[i] = depths[parent] + 1
            depths[start:end] = info_depths[i]
        self.layers = []
        for depth in sorted(set(info_depths)):
            groups = []
            for count in sorted(set(self.counts[info_depths == depth])):
                infosets = np.flatnonzero((info_depths == depth) & (self.counts == count))
                indices = self.starts[infosets, None] + np.arange(count)[None, :]
                groups.append((indices, self.parents[infosets]))
            self.layers.append(groups)

    def regret_matching(self, regrets):
        positive = np.maximum(regrets, 0.0)
        result = np.zeros_like(positive)
        result[0] = 1.0
        if self.starts.size:
            totals = np.add.reduceat(positive[1:], self.starts - 1)
            denom = totals[self.owner]
            np.divide(positive[1:], denom, out=result[1:], where=denom > 0)
            zero = denom <= 0
            result[1:][zero] = 1.0 / self.counts[self.owner[zero]]
        return result

    def sequence_form(self, behavioral):
        result = behavioral.copy()
        result[0] = 1.0
        for layer in self.layers:
            for indices, parents in layer:
                result[indices] *= result[parents, None]
        return result

    def best_response(self, reward):
        result = np.zeros(self.treeplex.num_sequences)
        result[0] = 1.0
        for layer in reversed(self.layers):
            for indices, parents in layer:
                action_values = reward[indices]
                choices = np.argmax(action_values, axis=1)
                chosen_sequences = indices[np.arange(len(indices)), choices]
                result[chosen_sequences] = 1.0
                np.add.at(reward, parents, reward[chosen_sequences])
        return result

    def accumulate_regret(self, reward, behavioral, regrets, plus):
        for layer in reversed(self.layers):
            for indices, parents in layer:
                action_values = reward[indices]
                value = np.sum(behavioral[indices] * action_values, axis=1)
                updated = regrets[indices] + action_values - value[:, None]
                regrets[indices] = np.maximum(updated, 0.0) if plus else updated
                np.add.at(reward, parents, value)


class Solver:
    def __init__(self, treeplex_game: TreeplexGame):
        self.game = treeplex_game
        self._ops = {
            PLAYER_1: _TreeplexOperations(self.game.treeplex_p1),
            PLAYER_2: _TreeplexOperations(self.game.treeplex_p2),
        }

    def compute_payoffs_from_seq_form_strategies(self, seq_form_strategy_p1, seq_form_strategy_p2):
        """Evaluate the supplied sequence-form strategies, from Player 1's view."""
        return float(sum(
            seq_form_strategy_p1[s1] * seq_form_strategy_p2[s2] * payoff
            for (s1, s2), payoff in self.game.seq_pair_to_leaves.items()
        ))

    def compute_reward_vector(self, player_id, opponent_seq_form_strategy):
        """Return own sequence rewards against a fixed opponent (P2 has -u1)."""
        if player_id not in (PLAYER_1, PLAYER_2):
            raise ValueError("player_id must be PLAYER_1 or PLAYER_2")
        treeplex = self._ops[player_id].treeplex
        opponent_treeplex = self._ops[1 - player_id].treeplex
        opponent = np.asarray(opponent_seq_form_strategy)
        if opponent.ndim != 1 or opponent.size != opponent_treeplex.num_sequences:
            raise ValueError("Opponent strategy size does not match their treeplex")
        reward = np.zeros(treeplex.num_sequences)
        for (s1, s2), payoff in self.game.seq_pair_to_leaves.items():
            if player_id == PLAYER_1:
                reward[s1] += opponent[s2] * payoff
            else:
                reward[s2] -= opponent[s1] * payoff
        return reward

    def compute_best_response(self, player_id, opponent_seq_form_strategy,
                              convert_to_seq_form: bool = False):
        """Find an exact deterministic behavioral (or sequence-form) response.

        At each infoset, choose the maximum continuation reward and add that
        reward to the parent sequence.  Chance and opponent reach weights are
        already included in the reward vector.
        """
        reward = self.compute_reward_vector(player_id, opponent_seq_form_strategy)
        ops = self._ops[player_id]
        behavioral = ops.best_response(reward)
        return ops.sequence_form(behavioral) if convert_to_seq_form else behavioral

    def saddle_point_residual(self, sol_p1, sol_p2):
        """Sum of both players' unilateral improvement opportunities."""
        br1 = self.compute_best_response(PLAYER_1, sol_p2, convert_to_seq_form=True)
        br2 = self.compute_best_response(PLAYER_2, sol_p1, convert_to_seq_form=True)
        upper = self.compute_payoffs_from_seq_form_strategies(br1, sol_p2)
        lower = self.compute_payoffs_from_seq_form_strategies(sol_p1, br2)
        # Mathematically nonnegative for feasible strategies; absorb only roundoff.
        gap = upper - lower
        return float(0.0 if -1e-12 < gap < 0.0 else gap)


class CFRSolver(Solver):
    """Full-tree CFR, with optional RM+, alternating updates and linear averaging.

    Defaults implement simultaneous vanilla CFR with uniform sequence averaging.
    ``regret_matching_plus`` floors cumulative regret *after* adding instantaneous
    regret: R <- max(R + r, 0).  ``averaging='linear'`` changes only the strategy
    average, not regret weights (so this option alone is not LCFR).

    With ``alternating=True``, P2 responds to P1's newly updated strategy within
    each iteration.  The averaged pair then consists of these sequentially played
    strategies.  ``averaging_delay`` omits the first N strategy pairs from the
    average; regret learning continues throughout the delay.
    """

    def __init__(self, treeplex_game: TreeplexGame, regret_matching_plus=False,
                 averaging="uniform", alternating=False, averaging_delay=0):
        super().__init__(treeplex_game)
        if averaging not in ("uniform", "linear"):
            raise ValueError("averaging must be 'uniform' or 'linear'")
        if not isinstance(averaging_delay, (int, np.integer)) or averaging_delay < 0:
            raise ValueError("averaging_delay must be a nonnegative integer")
        self.regret_matching_plus = bool(regret_matching_plus)
        self.averaging = averaging
        self.alternating = bool(alternating)
        self.averaging_delay = int(averaging_delay)
        self._reset()

    def _reset(self):
        self.regrets = {p: np.zeros(ops.treeplex.num_sequences) for p, ops in self._ops.items()}
        self.strategy_sums = {p: np.zeros(ops.treeplex.num_sequences) for p, ops in self._ops.items()}
        self.iterations = 0
        self.training_seconds = 0.0
        self.average_weight = 0.0
        self.history = []
        self.spr = float("inf")
        self.val = 0.0
        self._last_sequences = {
            p: ops.sequence_form(ops.regret_matching(self.regrets[p]))
            for p, ops in self._ops.items()
        }

    def _prepare_payoffs(self):
        # Snapshot each solve call so continuation can follow changed game payoffs.
        items = [(key, value) for key, value in self.game.seq_pair_to_leaves.items() if value != 0.0]
        self._edge_p1 = np.fromiter((key[0] for key, _ in items), dtype=np.intp, count=len(items))
        self._edge_p2 = np.fromiter((key[1] for key, _ in items), dtype=np.intp, count=len(items))
        self._edge_values = np.fromiter((value for _, value in items), dtype=float, count=len(items))

    def _reward(self, player, opponent):
        if player == PLAYER_1:
            return np.bincount(self._edge_p1,
                               weights=self._edge_values * opponent[self._edge_p2],
                               minlength=self.game.treeplex_p1.num_sequences)
        return np.bincount(self._edge_p2,
                           weights=-self._edge_values * opponent[self._edge_p1],
                           minlength=self.game.treeplex_p2.num_sequences)

    def _evaluate_average(self):
        if self.average_weight:
            x, y = (self.strategy_sums[p] / self.average_weight for p in (PLAYER_1, PLAYER_2))
        else:
            # Defined even when max_iter=0 or no samples survive averaging delay.
            x, y = (self._last_sequences[p].copy() for p in (PLAYER_1, PLAYER_2))
        self.sol_seq_form_p1, self.sol_seq_form_p2 = x, y
        br1 = self._ops[PLAYER_1].sequence_form(
            self._ops[PLAYER_1].best_response(self._reward(PLAYER_1, y)))
        br2 = self._ops[PLAYER_2].sequence_form(
            self._ops[PLAYER_2].best_response(self._reward(PLAYER_2, x)))
        upper = np.sum(self._edge_values * br1[self._edge_p1] * y[self._edge_p2])
        lower = np.sum(self._edge_values * x[self._edge_p1] * br2[self._edge_p2])
        self.spr = float(max(0.0, upper - lower))
        self.val = float(np.sum(self._edge_values * x[self._edge_p1] * y[self._edge_p2]))
        return x, y

    def solve(self, max_iter=1000, spr_threshold=1e-2, verbose: bool = False,
              reset: bool = False, *, check_every=100):
        """Take up to ``max_iter`` additional iterations and return average sequences.

        Existing regrets and average accumulators persist across calls unless
        ``reset=True``.  ``spr_threshold=None`` disables early stopping.  Exact
        best-response checks occur every ``check_every`` iterations and at the
        final iterate; the stopping decision uses the returned average strategy.
        """
        if not isinstance(max_iter, (int, np.integer)) or max_iter < 0:
            raise ValueError("max_iter must be a nonnegative integer")
        if not isinstance(check_every, (int, np.integer)) or check_every < 1:
            raise ValueError("check_every must be a positive integer")
        if spr_threshold is not None and spr_threshold < 0:
            raise ValueError("spr_threshold must be nonnegative or None")
        if reset:
            self._reset()
        self._prepare_payoffs()
        self._evaluate_average()
        # A repeated call may already satisfy the requested precision.
        if self.iterations and spr_threshold is not None and self.spr <= spr_threshold:
            return self.sol_seq_form_p1.copy(), self.sol_seq_form_p2.copy()
        for step in range(int(max_iter)):
            training_start = time.perf_counter()
            behavioral = {p: ops.regret_matching(self.regrets[p]) for p, ops in self._ops.items()}
            sequences = {p: self._ops[p].sequence_form(behavioral[p]) for p in self._ops}
            reward1 = self._reward(PLAYER_1, sequences[PLAYER_2])
            self._ops[PLAYER_1].accumulate_regret(
                reward1, behavioral[PLAYER_1], self.regrets[PLAYER_1], self.regret_matching_plus)
            if self.alternating:
                behavioral[PLAYER_1] = self._ops[PLAYER_1].regret_matching(self.regrets[PLAYER_1])
                sequences[PLAYER_1] = self._ops[PLAYER_1].sequence_form(behavioral[PLAYER_1])
            reward2 = self._reward(PLAYER_2, sequences[PLAYER_1])
            self._ops[PLAYER_2].accumulate_regret(
                reward2, behavioral[PLAYER_2], self.regrets[PLAYER_2], self.regret_matching_plus)
            self.iterations += 1
            average_index = self.iterations - self.averaging_delay
            if average_index > 0:
                weight = float(average_index if self.averaging == "linear" else 1)
                for p in self._ops:
                    self.strategy_sums[p] += weight * sequences[p]
                self.average_weight += weight
            self._last_sequences = sequences
            self.training_seconds += time.perf_counter() - training_start
            if self.iterations % check_every == 0 or step == max_iter - 1:
                self._evaluate_average()
                self.history.append({"iteration": self.iterations, "spr": self.spr, "value": self.val})
                if verbose:
                    print(f"iteration={self.iterations} SPR={self.spr:.8g} value={self.val:.8g}", flush=True)
                if self.average_weight and spr_threshold is not None and self.spr <= spr_threshold:
                    break
        self._evaluate_average()
        self.current_seq_form_p1 = self._ops[PLAYER_1].sequence_form(
            self._ops[PLAYER_1].regret_matching(self.regrets[PLAYER_1]))
        self.current_seq_form_p2 = self._ops[PLAYER_2].sequence_form(
            self._ops[PLAYER_2].regret_matching(self.regrets[PLAYER_2]))
        return self.sol_seq_form_p1.copy(), self.sol_seq_form_p2.copy()
