"""Simultaneous fictitious play over full-game sequence-form strategies."""

from operator import index
from time import perf_counter

import numpy as np

from env_config import PLAYER_1, PLAYER_2
from solver import Solver


class FictitiousPlaySolver(Solver):
    """Best-respond to the opponent's empirical mixture at each iteration.

    The initial belief is a uniform behavioral strategy, represented by one
    realization-plan observation. After t iterations the average is therefore
    (initial_plan + sum of t best-response plans) / (t + 1). Both best responses
    use the previous averages, so the update is simultaneous. A best response
    is a complete information-set-consistent plan, not an action chosen with
    access to an opponent's private cards.

    Topology stays fixed; payoff changes between solve calls are supported.
    reset=True discards empirical play from previous calls. This is ordinary
    fictitious play with a finite initial prior, without regret matching or LPs.
    """

    def __init__(self, treeplex_game, check_every=25):
        super().__init__(treeplex_game)
        self.check_every = index(check_every)
        if self.check_every < 1:
            raise ValueError("check_every must be positive")
        self._reset()

    def _reset(self):
        self.iterations = 0
        self.last_iterations = 0
        self.br_calls = 0
        self.diagnostic_br_calls = 0
        self.elapsed_seconds = 0.0
        self.last_elapsed_seconds = 0.0
        self.history = []
        self.spr = float("inf")
        self.val = float("nan")
        self.converged = False
        self._sum_p1 = self._operations[PLAYER_1].realization_plan(
            self._operations[PLAYER_1].uniform)
        self._sum_p2 = self._operations[PLAYER_2].realization_plan(
            self._operations[PLAYER_2].uniform)
        self._update_averages()
        self.current_seq_form_p1 = self.sol_seq_form_p1.copy()
        self.current_seq_form_p2 = self.sol_seq_form_p2.copy()

    @property
    def training_br_calls(self):
        """Count only best responses used to update empirical play."""
        return self.br_calls

    def _update_averages(self):
        # Convex mixtures belong in realization form; behavioral means lose own reach.
        denominator = self.iterations + 1
        self.sol_seq_form_p1 = self._sum_p1 / denominator
        self.sol_seq_form_p2 = self._sum_p2 / denominator

    def _record(self, started):
        self._update_averages()
        self.spr = self.saddle_point_residual(self.sol_seq_form_p1, self.sol_seq_form_p2)
        self.diagnostic_br_calls += 2
        self.val = self.compute_payoffs_from_seq_form_strategies(
            self.sol_seq_form_p1, self.sol_seq_form_p2)
        record = dict(iteration=self.iterations, spr=self.spr, value=self.val,
                      elapsed_seconds=self.elapsed_seconds + perf_counter() - started,
                      br_calls=self.br_calls, diagnostic_br_calls=self.diagnostic_br_calls)
        if self.history and self.history[-1]["iteration"] == self.iterations:
            self.history[-1] = record
        else:
            self.history.append(record)

    def solve(self, max_iter=2000, spr_threshold=0.002, verbose=False, reset=False):
        """Run up to max_iter additional simultaneous best-response iterations.

        Return the average realization plans. Check the full SPR initially,
        every check_every global iterations, and at return. None or a negative
        threshold disables early stopping. max_iter=0 evaluates the existing
        average. br_calls excludes the two diagnostic BRs per SPR evaluation;
        diagnostic_br_calls counts those separately. current_seq_form_p1/p2
        contain the most recently played pure best responses (or the seed).
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
            self._record(started)
            self.converged = bool(target_enabled and self.spr <= spr_threshold)
            for _ in range(max_iter):
                if self.converged:
                    break
                # Both beliefs precede either update: avoid turning this into alternating FP.
                first = self.compute_best_response(
                    PLAYER_1, self.sol_seq_form_p2, convert_to_seq_form=True)
                second = self.compute_best_response(
                    PLAYER_2, self.sol_seq_form_p1, convert_to_seq_form=True)
                self.br_calls += 2
                self._sum_p1 += first
                self._sum_p2 += second
                self.iterations += 1
                self.current_seq_form_p1, self.current_seq_form_p2 = first, second
                self._update_averages()
                if self.iterations % self.check_every == 0:
                    self._record(started)
                    self.converged = bool(target_enabled and self.spr <= spr_threshold)
                    if verbose:
                        print(f"iteration={self.iterations} SPR={self.spr:.8f} value={self.val:+.8f}")
            self._record(started)
            self.converged = bool(target_enabled and self.spr <= spr_threshold)
            if verbose:
                print(f"finished iteration={self.iterations} SPR={self.spr:.8f} value={self.val:+.8f}")
            return self.sol_seq_form_p1, self.sol_seq_form_p2
        finally:
            self._active_payoffs = None
            self.last_iterations = self.iterations - starting_iteration
            self.last_elapsed_seconds = perf_counter() - started
            self.elapsed_seconds += self.last_elapsed_seconds
