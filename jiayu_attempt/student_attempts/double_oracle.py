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
    sequence form. Added plans are deterministic best responses (pure, reduced
    strategies). The returned plans are the current restricted equilibrium, not
    a time average.

    Iteration t solves the restricted game, giving (x_t, y_t), and calls both
    global best-response oracles against it. Their values are an exact full-game
    certificate, SPR(x_t, y_t) = u(BR_1(y_t), y_t) - u(x_t, BR_2(x_t)), so the
    target is tested at every iteration without extra oracle calls. The
    restricted-game gap is ~0 by construction and is never used for stopping.
    If neither response is new, both lie in the restricted sets, which implies
    SPR(x_t, y_t) <= 0 up to LP tolerance: (x_t, y_t) is then an exact equilibrium
    of the full game and the solver stops ("no_new_strategy").
    """

    def __init__(self, treeplex_game, *, check_every=25):
        super().__init__(treeplex_game)
        # Kept for a common constructor; the oracle certifies every iteration.
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
        matrix = self.restricted_payoffs
        rows, columns = matrix.shape
        if rows == 1 and columns == 1:
            # A 1-by-1 game is its own equilibrium; no LP is needed.
            self.weights_p1, self.weights_p2 = np.ones(1), np.ones(1)
            self.sol_seq_form_p1 = self.support_p1[0].copy()
            self.sol_seq_form_p2 = self.support_p2[0].copy()
            self.restricted_gap = 0.0
            self._restricted_solved = True
            return
        try:
            from scipy.optimize import linprog
        except ImportError as exc:
            raise ImportError("DoubleOracleSolver requires scipy for restricted LPs.") from exc

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
        record = dict(
            iteration=self.iterations, spr=self.spr, value=self.val,
            elapsed_seconds=self.elapsed_seconds + perf_counter() - started,
            support_count_p1=self.support_count_p1, support_count_p2=self.support_count_p2,
            active_support_p1=int(np.count_nonzero(self.weights_p1 > 1e-12)),
            active_support_p2=int(np.count_nonzero(self.weights_p2 > 1e-12)),
            training_br_calls=self.training_br_calls,
            diagnostic_br_calls=self.diagnostic_br_calls,
            restricted_lp_solves=self.restricted_lp_solves,
        )
        if self.history and self.history[-1]["iteration"] == self.iterations:
            self.history[-1] = record
        else:
            self.history.append(record)

    def solve(self, max_iter=2000, spr_threshold=2e-3, verbose=False, reset=False):
        """Run up to max_iter additional oracle expansions; return the current plans.

        Every iteration certifies the current restricted equilibrium with the two
        oracle calls it needs anyway, and the run stops at the first certified
        SPR <= spr_threshold ("spr_threshold"), when neither oracle finds a new
        plan ("no_new_strategy": exact equilibrium up to LP tolerance), or after
        max_iter expansions ("iteration_limit"). None or a negative threshold
        disables the target. max_iter=0 certifies the current restricted
        equilibrium (re-solving it first if payoffs changed) without expanding.
        Continuation preserves restricted supports and mixtures; reset=True
        starts with uniform seeds. Fixed-topology payoff changes are refreshed
        at call boundaries, with the restricted equilibrium re-solved as needed.

        training_br_calls counts oracle calls whose responses were offered to the
        restricted game; diagnostic_br_calls counts the final two calls per
        solve() that only certified the returned profile. restricted_lp_solves
        counts each individual player LP (two per restricted_game_solves).
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
            while True:
                if not self._restricted_solved:
                    self._solve_restricted()
                x, y = self.sol_seq_form_p1, self.sol_seq_form_p2
                response1 = self.compute_best_response(PLAYER_1, y, convert_to_seq_form=True)
                upper = self.last_best_response_value           # max_x' u(x', y)
                response2 = self.compute_best_response(PLAYER_2, x, convert_to_seq_form=True)
                lower = -self.last_best_response_value          # min_y' u(x, y')
                # Oracle values give the exact full-game residual of (x, y).
                self.spr = float(upper - lower)
                self.val = self.compute_payoffs_from_seq_form_strategies(x, y)
                self.converged = bool(target_enabled and self.spr <= spr_threshold)
                self._record(started)
                if verbose:
                    print(f"iteration={self.iterations} SPR={self.spr:.8f} "
                          f"value={self.val:+.8f} supports={self.support_counts}")
                if self.converged:
                    self.stop_reason = "spr_threshold"
                elif self.iterations - starting_iteration >= max_iter:
                    self.stop_reason = "iteration_limit"
                elif not self._add_responses(response1, response2):
                    self.stop_reason = "no_new_strategy"
                else:
                    self.training_br_calls += 2
                    self.iterations += 1
                    continue
                self.diagnostic_br_calls += 2
                self._record(started)
                return self.sol_seq_form_p1, self.sol_seq_form_p2
        finally:
            self._active_payoffs = None
            self.last_iterations = self.iterations - starting_iteration
            self.last_elapsed_seconds = perf_counter() - started
            self.elapsed_seconds += self.last_elapsed_seconds
