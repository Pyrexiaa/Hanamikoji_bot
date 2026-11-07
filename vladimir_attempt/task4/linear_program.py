"""Independent sequence-form Nash equilibrium check using optional SciPy.

The assignment solver needs only NumPy. This experimental reference uses two
separate sparse linear programs, making it independent of CFR regret updates.
"""

import time

import numpy as np


def _flow_matrix(treeplex, sparse):
    """Return E, e for x >= 0, E x = e (including the empty sequence)."""
    rows = [0]
    cols = [0]
    values = [1.0]
    for index, (start, end, parent) in enumerate(zip(
            treeplex.info_state_start_seq, treeplex.info_state_end_seq,
            treeplex.info_state_parent_seq), start=1):
        rows.append(index)
        cols.append(parent)
        values.append(-1.0)
        rows.extend([index] * (end - start))
        cols.extend(range(start, end))
        values.extend([1.0] * (end - start))
    matrix = sparse.coo_matrix(
        (values, (rows, cols)),
        shape=(treeplex.get_num_infosets() + 1, treeplex.num_sequences),
    ).tocsr()
    rhs = np.zeros(matrix.shape[0])
    rhs[0] = 1.0
    return matrix, rhs


def solve_sequence_form_lp(game):
    """Return player realization plans ``(x, y, metadata)`` via sparse LPs.

    With E x = e and F y = f, Player 1 solves
    max f.T q subject to F.T q <= A.T x, x >= 0, q unrestricted.
    Player 2 independently solves
    min e.T p subject to A y <= E.T p, y >= 0, p unrestricted.

    SciPy is imported only when this reference method is requested. Returned
    plans use sequence form; use the provided conversion before saving agents.
    """
    from scipy import sparse
    from scipy.optimize import linprog

    started = time.perf_counter()
    tp1, tp2 = game.treeplex_p1, game.treeplex_p2
    e_matrix, e = _flow_matrix(tp1, sparse)
    f_matrix, f = _flow_matrix(tp2, sparse)
    entries = [(i, j, value) for (i, j), value
               in game.seq_pair_to_leaves.items() if value != 0.0]
    if entries:
        row, col, values = zip(*entries)
    else:
        row, col, values = [], [], []
    payoff = sparse.coo_matrix(
        (values, (row, col)),
        shape=(tp1.num_sequences, tp2.num_sequences),
    ).tocsr()
    build_seconds = time.perf_counter() - started
    options = {
        "primal_feasibility_tolerance": 1e-9,
        "dual_feasibility_tolerance": 1e-9,
        "ipm_optimality_tolerance": 1e-10,
    }

    p1_started = time.perf_counter()
    maximin = linprog(
        np.concatenate((np.zeros(tp1.num_sequences), -f)),
        A_ub=sparse.hstack((-payoff.T, f_matrix.T), format="csr"),
        b_ub=np.zeros(tp2.num_sequences),
        A_eq=sparse.hstack((e_matrix, sparse.csr_matrix(
            (e_matrix.shape[0], f_matrix.shape[0]))), format="csr"),
        b_eq=e,
        bounds=([(0, None)] * tp1.num_sequences
                + [(None, None)] * f_matrix.shape[0]),
        method="highs",
        options=options,
    )
    p1_seconds = time.perf_counter() - p1_started
    if not maximin.success:
        raise RuntimeError("Player 1 maximin LP failed: " + maximin.message)

    p2_started = time.perf_counter()
    minimax = linprog(
        np.concatenate((np.zeros(tp2.num_sequences), e)),
        A_ub=sparse.hstack((payoff, -e_matrix.T), format="csr"),
        b_ub=np.zeros(tp1.num_sequences),
        A_eq=sparse.hstack((f_matrix, sparse.csr_matrix(
            (f_matrix.shape[0], e_matrix.shape[0]))), format="csr"),
        b_eq=f,
        bounds=([(0, None)] * tp2.num_sequences
                + [(None, None)] * e_matrix.shape[0]),
        method="highs",
        options=options,
    )
    p2_seconds = time.perf_counter() - p2_started
    if not minimax.success:
        raise RuntimeError("Player 2 minimax LP failed: " + minimax.message)

    x = maximin.x[:tp1.num_sequences].copy()
    y = minimax.x[:tp2.num_sequences].copy()
    metadata = {
        "maximin_value": float(-maximin.fun),
        "minimax_value": float(minimax.fun),
        "duality_gap": float(minimax.fun + maximin.fun),
        "profile_payoff": float(x @ (payoff @ y)),
        "max_flow_error_p1": float(np.max(np.abs(e_matrix @ x - e))),
        "max_flow_error_p2": float(np.max(np.abs(f_matrix @ y - f))),
        "min_probability_p1": float(x.min()),
        "min_probability_p2": float(y.min()),
        "nonzero_payoff_pairs": int(payoff.nnz),
        "build_seconds": build_seconds,
        "p1_solve_seconds": p1_seconds,
        "p2_solve_seconds": p2_seconds,
        "solve_seconds": p1_seconds + p2_seconds,
        "total_seconds": time.perf_counter() - started,
        "p1_lp_iterations": int(maximin.nit),
        "p2_lp_iterations": int(minimax.nit),
    }
    return x, y, metadata
