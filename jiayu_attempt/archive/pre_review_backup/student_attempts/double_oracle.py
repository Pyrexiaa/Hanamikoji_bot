"""Double oracle over restricted sets of sequence-form strategies.

Only the restricted normal-form matrix is dense. Its equilibrium mixtures are
combined in realization-plan space, preserving all sequence-flow constraints.
SciPy is imported lazily when a restricted LP actually needs solving.
"""

from operator import index
from time import perf_counter

import numpy as np

from env_config import PLAYER_1, PLAYER_2
from solver import Solver


class DoubleOracleSolver(Solver):
    """Expand a restricted zero-sum game with both players' global best responses.

    Each restricted set starts with the uniform behavioral strategy converted to
    sequence form. Added plans are deterministic best responses. The returned
    plans are the current restricted equilibrium, not a time average.

    One iteration calls both global best-response oracles, adds any new plans,
    and solves the expanded restricted game. A repeated oracle plan is ignored.
    Even when neither set grows, iterations continue to the next scheduled full
    SPR check; a restricted-game gap never controls early stopping.
    """

    def __init__(self, treeplex_game, *, check_every=25):
        super().__init__(treeplex_game)
        self.check_every = index(check_every)
        if self.check_every < 1:
            raise ValueError("check_every must be positive")
        self._reset()

    @property
    def support_count_p1(self):
        return len(self.support_p1)

    @property
    def support_count_p2(self):
        return len(self.support_p2)

    @property
    def support_counts(self):
        return self.support_count_p1, self.support_count_p2

    def _reset(self):
        self.sol_seq_form_p1 = self._operations[PLAYER_1].realization_plan(
            self._operations[PLAYER_1].uniform)
        self.sol_seq_form_p2 = self._operations[PLAYER_2].realization_plan(
            self._operations[PLAYER_2].uniform)
        self.support_p1 = [self.sol_seq_form_p1.copy()]
        self.support_p2 = [self.sol_seq_form_p2.copy()]
        self._keys_p1 = {self.support_p1[0].tobytes()}
        self._keys_p2 = {self.support_p2[0].tobytes()}
        self.weights_p1 = np.ones(1)
        self.weights_p2 = np.ones(1)
        self.restricted_payoffs = None
        self.restricted_gap = float("nan")
        self._restricted_solved = False
        self._previous_payoffs = None
        self.iterations = 0
        self.last_iterations = 0
        self.training_br_calls = 0
        self.diagnostic_br_calls = 0
        self.restricted_lp_solves = 0
        self.restricted_game_solves = 0
        self.elapsed_seconds = 0.0
        self.last_elapsed_seconds = 0.0
        self.history = []
        self.spr = float("inf")
        self.val = float("nan")
        self.converged = False
        self.stop_reason = None

    def _refresh_restricted_payoffs(self):
        current = self._active_payoffs
        changed = self._previous_payoffs is None or any(
            not np.array_equal(old, new)
            for old, new in zip(self._previous_payoffs or (), current)
        )
        if not changed:
            return
        plans_p1 = np.asarray(self.support_p1)
        self.restricted_payoffs = np.column_stack([
            plans_p1 @ self.compute_reward_vector(PLAYER_1, y)
            for y in self.support_p2
        ])
        self._previous_payoffs = tuple(array.copy() for array in current)
        self._restricted_solved = False

    def _add_responses(self, response_p1, response_p2):
        old_rows, old_columns = self.restricted_payoffs.shape
        key1, key2 = response_p1.tobytes(), response_p2.tobytes()
        new1, new2 = key1 not in self._keys_p1, key2 not in self._keys_p2
        if not (new1 or new2):
            return False
        if new1:
            self.support_p1.append(response_p1.copy())
            self._keys_p1.add(key1)
        if new2:
            self.support_p2.append(response_p2.copy())
            self._keys_p2.add(key2)
        expanded = np.empty(self.support_counts)
        expanded[:old_rows, :old_columns] = self.restricted_payoffs
        if new1:
            reward = -self.compute_reward_vector(PLAYER_2, response_p1)
            expanded[-1, :old_columns] = np.asarray(self.support_p2[:old_columns]) @ reward
        if new2:
            reward = self.compute_reward_vector(PLAYER_1, response_p2)
            expanded[:, -1] = np.asarray(self.support_p1) @ reward
        self.restricted_payoffs = expanded
        self._restricted_solved = False
        return True

    def _solve_restricted(self):
        try:
            from scipy.optimize import linprog
        except ImportError as exc:
            raise ImportError("DoubleOracleSolver requires scipy for restricted LPs.") from exc

        matrix = self.restricted_payoffs
        rows, columns = matrix.shape
        options = {"primal_feasibility_tolerance": 1e-9,
                   "dual_feasibility_tolerance": 1e-9}
        # The game value is free, including for negative-payoff games.
        first = linprog(
            np.r_[np.zeros(rows), -1.0],
            A_ub=np.column_stack([-matrix.T, np.ones(columns)]),
            b_ub=np.zeros(columns),
            A_eq=np.r_[np.ones(rows), 0.0][None, :],
            b_eq=np.ones(1),
            bounds=[(0.0, None)] * rows + [(None, None)],
            method="highs", options=options,
        )
        self.restricted_lp_solves += 1
        if not first.success:
            raise RuntimeError(f"Player-one restricted LP failed: {first.message}")
        second = linprog(
            np.r_[np.zeros(columns), 1.0],
            A_ub=np.column_stack([matrix, -np.ones(rows)]),
            b_ub=np.zeros(rows),
            A_eq=np.r_[np.ones(columns), 0.0][None, :],
            b_eq=np.ones(1),
            bounds=[(0.0, None)] * columns + [(None, None)],
            method="highs", options=options,
        )
        self.restricted_lp_solves += 1
        if not second.success:
            raise RuntimeError(f"Player-two restricted LP failed: {second.message}")

        self.weights_p1 = np.maximum(first.x[:rows], 0.0)
        self.weights_p2 = np.maximum(second.x[:columns], 0.0)
        self.weights_p1 /= self.weights_p1.sum()
        self.weights_p2 /= self.weights_p2.sum()
        # Mixing behavioral coordinates would lose correlations with own reach.
        self.sol_seq_form_p1 = self.weights_p1 @ np.asarray(self.support_p1)
        self.sol_seq_form_p2 = self.weights_p2 @ np.asarray(self.support_p2)
        self.restricted_gap = max(0.0, float(second.fun + first.fun))
        self.restricted_game_solves += 1
        self._restricted_solved = True

    def _record(self, started):
        self.spr = self.saddle_point_residual(self.sol_seq_form_p1, self.sol_seq_form_p2)
        self.diagnostic_br_calls += 2
        self.val = self.compute_payoffs_from_seq_form_strategies(
            self.sol_seq_form_p1, self.sol_seq_form_p2)
        record = dict(
            iteration=self.iterations, spr=self.spr, value=self.val,
            elapsed_seconds=self.elapsed_seconds + perf_counter() - started,
            support_count_p1=self.support_count_p1, support_count_p2=self.support_count_p2,
            training_br_calls=self.training_br_calls,
            diagnostic_br_calls=self.diagnostic_br_calls,
            restricted_lp_solves=self.restricted_lp_solves,
        )
        if self.history and self.history[-1]["iteration"] == self.iterations:
            self.history[-1] = record
        else:
            self.history.append(record)

    def solve(self, max_iter=2000, spr_threshold=2e-3, verbose=False, reset=False):
        """Run additional oracle iterations and return the current realization plans.

        Convergence uses the unrestricted saddle-point residual at entry, exit,
        and every check_every global iterations. None or a negative threshold
        disables stopping. max_iter=0 only evaluates the existing profile.
        Continuation preserves restricted supports and mixtures; reset=True
        starts with uniform seeds. Fixed-topology payoff changes are refreshed
        at call boundaries, with the restricted equilibrium re-solved as needed.

        training_br_calls counts expansion oracles, diagnostic_br_calls counts
        the two oracles per certificate, and restricted_lp_solves counts each
        individual player LP (two per restricted_game_solves).
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
        target_enabled = spr_threshold is not None and spr_threshold >= 0.0
        try:
            self._refresh_restricted_payoffs()
            self._record(started)
            self.converged = bool(target_enabled and self.spr <= spr_threshold)
            if max_iter and not self.converged and not self._restricted_solved:
                self._solve_restricted()
            for _ in range(max_iter):
                if self.converged:
                    break
                response1 = self.compute_best_response(
                    PLAYER_1, self.sol_seq_form_p2, convert_to_seq_form=True)
                response2 = self.compute_best_response(
                    PLAYER_2, self.sol_seq_form_p1, convert_to_seq_form=True)
                self.training_br_calls += 2
                if self._add_responses(response1, response2):
                    self._solve_restricted()
                self.iterations += 1
                if self.iterations % self.check_every == 0:
                    self._record(started)
                    self.converged = bool(target_enabled and self.spr <= spr_threshold)
                    if verbose:
                        print(f"iteration={self.iterations} SPR={self.spr:.8f} "
                              f"value={self.val:+.8f} supports={self.support_counts}")
            self._record(started)
            self.converged = bool(target_enabled and self.spr <= spr_threshold)
            self.stop_reason = "spr_threshold" if self.converged else "iteration_limit"
            if verbose:
                print(f"finished iteration={self.iterations} SPR={self.spr:.8f} "
                      f"value={self.val:+.8f} supports={self.support_counts}")
            return self.sol_seq_form_p1, self.sol_seq_form_p2
        finally:
            self._active_payoffs = None
            self.last_iterations = self.iterations - starting_iteration
            self.last_elapsed_seconds = perf_counter() - started
            self.elapsed_seconds += self.last_elapsed_seconds
