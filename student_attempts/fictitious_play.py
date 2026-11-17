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

    The two best responses of iteration t answer the current averages, so their
    values also certify those averages exactly:
    SPR(x_t, y_t) = u(BR_1(y_t), y_t) - u(x_t, BR_2(x_t)).
    The stopping rule therefore tests the target at every iteration without
    extra best-response calls; check_every only thins the recorded history.

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
        record = dict(iteration=self.iterations, spr=self.spr, value=self.val,
                      elapsed_seconds=self.elapsed_seconds + perf_counter() - started,
                      br_calls=self.br_calls, diagnostic_br_calls=self.diagnostic_br_calls)
        if self.history and self.history[-1]["iteration"] == self.iterations:
            self.history[-1] = record
        else:
            self.history.append(record)

    def solve(self, max_iter=2000, spr_threshold=0.002, verbose=False, reset=False):
        """Run up to max_iter additional simultaneous best-response iterations.

        Return the average realization plans. The exact SPR of the current
        averages is obtained at every iteration from the two best responses that
        the next update needs anyway, so the run stops at the first iteration
        whose averages meet the target. None or a negative threshold disables
        early stopping. max_iter=0 only certifies the existing average.

        br_calls counts the best responses that updated empirical play;
        diagnostic_br_calls counts the two responses per call whose only use was
        certifying the returned profile. History holds the entry profile, every
        check_every-th iteration, and the returned profile. current_seq_form_p1/p2
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
            while True:
                self._update_averages()
                x, y = self.sol_seq_form_p1, self.sol_seq_form_p2
                # Both beliefs precede either update: avoid turning this into alternating FP.
                first = self.compute_best_response(PLAYER_1, y, convert_to_seq_form=True)
                upper = self.last_best_response_value           # max_x' u(x', y)
                second = self.compute_best_response(PLAYER_2, x, convert_to_seq_form=True)
                lower = -self.last_best_response_value          # min_y' u(x, y')
                # Best responses to the current averages: an exact, free certificate.
                self.spr = float(upper - lower)
                self.converged = bool(target_enabled and self.spr <= spr_threshold)
                stopping = self.converged or self.iterations - starting_iteration >= max_iter
                if stopping:
                    self.diagnostic_br_calls += 2
                if (stopping or self.iterations == starting_iteration
                        or self.iterations % self.check_every == 0):
                    self.val = self.compute_payoffs_from_seq_form_strategies(x, y)
                    self._record(started)
                    if verbose:
                        print(f"iteration={self.iterations} SPR={self.spr:.8f} value={self.val:+.8f}")
                if stopping:
                    break
                self.br_calls += 2
                self._sum_p1 += first
                self._sum_p2 += second
                self.iterations += 1
                self.current_seq_form_p1, self.current_seq_form_p2 = first, second
            return self.sol_seq_form_p1, self.sol_seq_form_p2
        finally:
            self._active_payoffs = None
            self.last_iterations = self.iterations - starting_iteration
            self.last_elapsed_seconds = perf_counter() - started
            self.elapsed_seconds += self.last_elapsed_seconds
