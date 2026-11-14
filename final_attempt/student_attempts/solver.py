"""Sequence-form best responses and counterfactual regret minimization."""

from operator import index
from time import perf_counter

import numpy as np

from env_config import PLAYER_1, PLAYER_2
from treeplex_representation import TreeplexGame


class _TreeplexOperations:
    """Cache fixed topology; batch independent information sets by own-action depth."""

    def __init__(self, treeplex):
        self.size = treeplex.num_sequences
        starts = np.asarray(treeplex.info_state_start_seq, dtype=np.intp)
        ends = np.asarray(treeplex.info_state_end_seq, dtype=np.intp)
        parents = np.asarray(treeplex.info_state_parent_seq, dtype=np.intp)
        self.counts = ends - starts
        self.num_infosets = len(starts)
        self.owners = np.repeat(np.arange(self.num_infosets), self.counts)
        self.uniform = np.ones(self.size)
        if self.num_infosets:
            self.uniform[1:] = 1.0 / self.counts[self.owners]

        sequence_depth = np.zeros(self.size, dtype=np.intp)
        depth_to_infosets = {}
        for info, (start, end, parent) in enumerate(zip(starts, ends, parents)):
            depth = int(sequence_depth[parent])
            depth_to_infosets.setdefault(depth, []).append(info)
            sequence_depth[start:end] = depth + 1

        self.levels = []
        for depth in sorted(depth_to_infosets):
            infos = np.asarray(depth_to_infosets[depth], dtype=np.intp)
            counts = self.counts[infos]
            local_owners = np.repeat(np.arange(len(infos)), counts)
            padded_actions = starts[infos, None] + np.arange(counts.max())[None, :]
            valid = np.arange(counts.max())[None, :] < counts[:, None]
            padded_actions[~valid] = 0
            actions = padded_actions[valid]
            self.levels.append((actions, local_owners, parents[infos], padded_actions, valid))

    def regret_matching(self, regrets):
        behavior = np.maximum(regrets, 0.0)
        totals = np.bincount(self.owners, weights=behavior[1:], minlength=self.num_infosets)
        denominator = totals[self.owners]
        behavior[1:] = np.divide(
            behavior[1:], denominator, out=self.uniform[1:].copy(), where=denominator > 0.0
        )
        behavior[0] = 1.0
        return behavior

    def realization_plan(self, behavior):
        sequence = behavior.copy()
        sequence[0] = 1.0
        for actions, owners, parents, _, _ in self.levels:
            sequence[actions] *= sequence[parents[owners]]
        return sequence

    def best_response(self, rewards):
        behavior = np.zeros(self.size)
        behavior[0] = 1.0
        for _, _, parents, padded, valid in reversed(self.levels):
            scores = np.where(valid, rewards[padded], -np.inf)
            best = padded[np.arange(len(parents)), np.argmax(scores, axis=1)]
            behavior[best] = 1.0
            # Several information sets can share a parent sequence: all values contribute.
            # Add them in descending information-set order, exactly as the sequential
            # reversed-order pass does. Floating-point addition is not associative, so
            # this keeps parent values bit-identical to that pass and resolves exact
            # ties (e.g. symmetric G1/G2 or identical split cards) the same way.
            np.add.at(rewards, parents[::-1], rewards[best][::-1])
        return behavior

    def update_regrets(self, regrets, rewards, behavior, regret_matching_plus):
        for actions, owners, parents, _, _ in reversed(self.levels):
            action_values = rewards[actions]
            values = np.bincount(
                owners, weights=behavior[actions] * action_values, minlength=len(parents)
            )
            # Opponent and chance reach are in rewards; own reach must not enter regret.
            regrets[actions] += action_values - values[owners]
            np.add.at(rewards, parents, values)
        if regret_matching_plus:
            np.maximum(regrets, 0.0, out=regrets)


class Solver:
    def __init__(self, treeplex_game: TreeplexGame):
        self.game = treeplex_game
        self._operations = {
            PLAYER_1: _TreeplexOperations(treeplex_game.treeplex_p1),
            PLAYER_2: _TreeplexOperations(treeplex_game.treeplex_p2),
        }
        self._active_payoffs = None
        self.last_best_response_value = float("nan")

    def _payoff_arrays(self):
        if self._active_payoffs is not None:
            return self._active_payoffs
        entries = self.game.seq_pair_to_leaves
        if not entries:
            return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp), np.empty(0)
        coordinates = np.asarray(list(entries), dtype=np.intp)
        values = np.fromiter(entries.values(), dtype=float, count=len(entries))
        return coordinates[:, 0], coordinates[:, 1], values

    def _strategy_array(self, strategy, player_id):
        if player_id not in self._operations:
            raise ValueError(f"Invalid player_id: {player_id}")
        strategy = np.asarray(strategy, dtype=float)
        if strategy.shape != (self._operations[player_id].size,):
            raise ValueError(f"Strategy size does not match player {player_id}'s treeplex")
        if not np.all(np.isfinite(strategy)):
            raise ValueError("Strategy contains a non-finite probability")
        return strategy

    def compute_payoffs_from_seq_form_strategies(self,
                                                 seq_form_strategy_p1: np.ndarray,
                                                 seq_form_strategy_p2: np.ndarray):
        """Evaluate x^T A y; the sparse payoffs already include chance probabilities."""
        x = self._strategy_array(seq_form_strategy_p1, PLAYER_1)
        y = self._strategy_array(seq_form_strategy_p2, PLAYER_2)
        rows, columns, payoffs = self._payoff_arrays()
        return float(np.sum(x[rows] * y[columns] * payoffs))

    def compute_reward_vector(self,
                              player_id: int,
                              opponent_seq_form_strategy: np.ndarray) -> np.ndarray:
        """Return A y for P1 or -A^T x for P2, with no own-player reach factor."""
        if player_id not in self._operations:
            raise ValueError(f"Invalid player_id: {player_id}")
        opponent_id = PLAYER_2 if player_id == PLAYER_1 else PLAYER_1
        opponent = self._strategy_array(opponent_seq_form_strategy, opponent_id)
        rows, columns, payoffs = self._payoff_arrays()
        if player_id == PLAYER_1:
            return np.bincount(rows, weights=opponent[columns] * payoffs,
                               minlength=self._operations[PLAYER_1].size)
        return np.bincount(columns, weights=-opponent[rows] * payoffs,
                           minlength=self._operations[PLAYER_2].size)

    def compute_best_response(self,
                              player_id: int,
                              opponent_seq_form_strategy: np.ndarray,
                              convert_to_seq_form: bool = False) -> np.ndarray:
        """Maximize the player's reward by bottom-up treeplex dynamic programming.

        The default result is a deterministic behavioral strategy. Ties select the
        first legal action; unreachable information sets still have legal distributions.
        Values are accumulated in the same order as the sequential reversed-infoset pass,
        so the returned vector is bit-identical to that reference implementation.
        """
        rewards = self.compute_reward_vector(player_id, opponent_seq_form_strategy)
        operations = self._operations[player_id]
        behavior = operations.best_response(rewards)
        # The bottom-up pass leaves the best-response value on the empty sequence
        # (in this player's reward units: P2's is minus P1's payoff). Iterative
        # solvers reuse it as a free certificate instead of re-evaluating payoffs.
        self.last_best_response_value = float(rewards[0])
        return operations.realization_plan(behavior) if convert_to_seq_form else behavior

    def saddle_point_residual(self, sol_p1: np.ndarray, sol_p2: np.ndarray):
        """Return max_x u(x, sol_p2) - min_y u(sol_p1, y), without a factor of 1/2."""
        br_p1 = self.compute_best_response(PLAYER_1, sol_p2, convert_to_seq_form=True)
        br_p2 = self.compute_best_response(PLAYER_2, sol_p1, convert_to_seq_form=True)
        upper = self.compute_payoffs_from_seq_form_strategies(br_p1, sol_p2)
        lower = self.compute_payoffs_from_seq_form_strategies(sol_p1, br_p2)
        return float(upper - lower)


class CFRSolver(Solver):
    """Full-tree CFR, optionally using RM+, alternating updates, and linear weights.

    Defaults implement simultaneous vanilla CFR. Each iteration averages both
    pre-update realization plans. Alternating updates first update P1's regrets,
    then update P2 against P1's refreshed strategy. The same averaging convention
    is used in either mode. Linear weights are the global iteration numbers.

    Topology must stay fixed. Payoff changes between solve calls are supported;
    reset=True discards regrets and historical averages from the previous game.
    """

    def __init__(self, treeplex_game: TreeplexGame, *, regret_matching_plus=False,
                 linear_averaging=False, alternating=False, check_every=25):
        super().__init__(treeplex_game)
        self.regret_matching_plus = bool(regret_matching_plus)
        self.linear_averaging = bool(linear_averaging)
        self.alternating = bool(alternating)
        self.check_every = index(check_every)
        if self.check_every < 1:
            raise ValueError("check_every must be positive")
        self._reset()

    def _reset(self):
        self.regrets_p1 = np.zeros(self._operations[PLAYER_1].size)
        self.regrets_p2 = np.zeros(self._operations[PLAYER_2].size)
        self._sum_p1 = np.zeros_like(self.regrets_p1)
        self._sum_p2 = np.zeros_like(self.regrets_p2)
        self._total_weight = 0.0
        self.iterations = 0
        self.last_iterations = 0
        self.elapsed_seconds = 0.0
        self.last_elapsed_seconds = 0.0
        self.history = []
        self.spr = float("inf")
        self.val = float("nan")
        self.converged = False
        self.sol_seq_form_p1 = self._operations[PLAYER_1].realization_plan(
            self._operations[PLAYER_1].uniform)
        self.sol_seq_form_p2 = self._operations[PLAYER_2].realization_plan(
            self._operations[PLAYER_2].uniform)
        self.current_seq_form_p1 = self.sol_seq_form_p1.copy()
        self.current_seq_form_p2 = self.sol_seq_form_p2.copy()
        self.last_played_seq_form_p1 = self.sol_seq_form_p1.copy()
        self.last_played_seq_form_p2 = self.sol_seq_form_p2.copy()

    def _record(self, started):
        if self._total_weight:
            self.sol_seq_form_p1 = self._sum_p1 / self._total_weight
            self.sol_seq_form_p2 = self._sum_p2 / self._total_weight
        self.spr = self.saddle_point_residual(self.sol_seq_form_p1, self.sol_seq_form_p2)
        self.val = self.compute_payoffs_from_seq_form_strategies(
            self.sol_seq_form_p1, self.sol_seq_form_p2)
        record = dict(iteration=self.iterations, spr=self.spr, value=self.val,
                      elapsed_seconds=self.elapsed_seconds + perf_counter() - started)
        if self.history and self.history[-1]["iteration"] == self.iterations:
            self.history[-1] = record
        else:
            self.history.append(record)

    def solve(self, max_iter=1000, spr_threshold=1e-2, verbose: bool = False, reset: bool = False):
        """Return the reach-weighted time-average sequence-form strategy profile.

        max_iter counts additional iterations. None or a negative SPR threshold
        disables early stopping. Convergence is checked initially, every check_every
        global iterations, and at return. max_iter=0 evaluates the existing average
        (uniform on a fresh solver). Returned arrays also populate sol_seq_form_p1/p2.
        history records the average's full SPR, P1 value, and cumulative solve time.
        """
        max_iter = index(max_iter)
        if max_iter < 0:
            raise ValueError("max_iter must be non-negative")
        if spr_threshold is not None and not np.isfinite(spr_threshold):
            raise ValueError("spr_threshold must be finite or None")
        if reset:
            self._reset()
        started = perf_counter()
        starting_iteration = self.iterations
        self._active_payoffs = self._payoff_arrays()
        op1, op2 = self._operations[PLAYER_1], self._operations[PLAYER_2]
        target_enabled = spr_threshold is not None and spr_threshold >= 0.0
        try:
            self._record(started)
            self.converged = bool(target_enabled and self.spr <= spr_threshold)
            for _ in range(max_iter):
                if self.converged:
                    break
                beh1 = op1.regret_matching(self.regrets_p1)
                beh2 = op2.regret_matching(self.regrets_p2)
                seq1, seq2 = op1.realization_plan(beh1), op2.realization_plan(beh2)
                self.iterations += 1
                weight = self.iterations if self.linear_averaging else 1.0
                # Averaging realization plans includes own reach at every information set.
                self._sum_p1 += weight * seq1
                self._sum_p2 += weight * seq2
                self._total_weight += weight
                self.last_played_seq_form_p1 = seq1
                self.last_played_seq_form_p2 = seq2

                rewards1 = self.compute_reward_vector(PLAYER_1, seq2)
                op1.update_regrets(self.regrets_p1, rewards1, beh1, self.regret_matching_plus)
                opponent_for_p2 = seq1
                if self.alternating:
                    opponent_for_p2 = op1.realization_plan(op1.regret_matching(self.regrets_p1))
                rewards2 = self.compute_reward_vector(PLAYER_2, opponent_for_p2)
                op2.update_regrets(self.regrets_p2, rewards2, beh2, self.regret_matching_plus)

                if self.iterations % self.check_every == 0:
                    self._record(started)
                    self.converged = bool(target_enabled and self.spr <= spr_threshold)
                    if verbose:
                        print(f"iteration={self.iterations} SPR={self.spr:.8f} value={self.val:+.8f}")
            self._record(started)
            self.converged = bool(target_enabled and self.spr <= spr_threshold)
            self.current_seq_form_p1 = op1.realization_plan(op1.regret_matching(self.regrets_p1))
            self.current_seq_form_p2 = op2.realization_plan(op2.regret_matching(self.regrets_p2))
            if verbose:
                print(f"finished iteration={self.iterations} SPR={self.spr:.8f} value={self.val:+.8f}")
            return self.sol_seq_form_p1, self.sol_seq_form_p2
        finally:
            self._active_payoffs = None
            self.last_iterations = self.iterations - starting_iteration
            self.last_elapsed_seconds = perf_counter() - started
            self.elapsed_seconds += self.last_elapsed_seconds
