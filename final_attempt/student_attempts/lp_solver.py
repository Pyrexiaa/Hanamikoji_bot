"""Optional sparse sequence-form LP reference for checking iterative solutions.

SciPy is only needed when these helpers are called; importing the CFR solver
or running an agent never imports this optional dependency.
"""

import numpy as np


def _scipy():
    try:
        from scipy import sparse
        from scipy.optimize import linprog
    except ImportError as exc:
        raise ImportError("The optional LP reference requires scipy.") from exc
    return sparse, linprog


def sequence_form_constraints(treeplex):
    """Return sparse E and e for the realization-plan constraints Ex = e."""
    sparse, _ = _scipy()
    rows, cols, data = [0], [0], [1.0]
    for index, (start, end, parent) in enumerate(zip(
        treeplex.info_state_start_seq,
        treeplex.info_state_end_seq,
        treeplex.info_state_parent_seq,
    ), start=1):
        rows.append(index)
        cols.append(parent)
        data.append(-1.0)
        rows.extend([index] * (end - start))
        cols.extend(range(start, end))
        data.extend([1.0] * (end - start))
    shape = (treeplex.get_num_infosets() + 1, treeplex.num_sequences)
    matrix = sparse.coo_matrix((data, (rows, cols)), shape=shape).tocsr()
    rhs = np.zeros(shape[0])
    rhs[0] = 1.0
    return matrix, rhs


def payoff_matrix(game):
    """Return A with player-one payoff x.T @ A @ y, including chance weights."""
    sparse, _ = _scipy()
    entries = list(game.seq_pair_to_leaves.items())
    rows = [pair[0] for pair, _ in entries]
    cols = [pair[1] for pair, _ in entries]
    values = [value for _, value in entries]
    matrix = sparse.coo_matrix(
        (values, (rows, cols)),
        shape=(game.treeplex_p1.num_sequences, game.treeplex_p2.num_sequences),
        dtype=float,
    ).tocsr()
    matrix.eliminate_zeros()
    return matrix


def _max_violation(matrix, rhs, plan):
    return max(float(np.max(np.abs(matrix @ plan - rhs))),
               max(0.0, -float(np.min(plan))))


def solve_sequence_form_lp(game, *, tolerance=1e-9):
    """Solve both players' sparse LPs and return plans plus certificates.

    With Ex=e and Fy=f, player one maximizes f.T q subject to
    F.T q <= A.T x; player two minimizes e.T p subject to E.T p >= A y.
    The dual variables p and q are unrestricted. Their objectives provide
    certified upper/lower bounds, up to the reported numerical residuals.
    """
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and strictly positive")
    sparse, linprog = _scipy()
    E, e = sequence_form_constraints(game.treeplex_p1)
    F, f = sequence_form_constraints(game.treeplex_p2)
    A = payoff_matrix(game)
    nx, ny = A.shape
    ne, nf = E.shape[0], F.shape[0]
    options = {
        "primal_feasibility_tolerance": max(float(tolerance), 1e-10),
        "dual_feasibility_tolerance": max(float(tolerance), 1e-10),
        "ipm_optimality_tolerance": max(float(tolerance), 1e-12),
    }

    # Minimizing the negative objective implements player one's maximization.
    result_p1 = linprog(
        np.r_[np.zeros(nx), -f],
        A_ub=sparse.hstack([-A.T, F.T], format="csr"),
        b_ub=np.zeros(ny),
        A_eq=sparse.hstack([E, sparse.csr_matrix((ne, nf))], format="csr"),
        b_eq=e,
        bounds=[(0, None)] * nx + [(None, None)] * nf,
        method="highs",
        options=options,
    )
    if not result_p1.success:
        raise RuntimeError(f"Player-one sequence-form LP failed: {result_p1.message}")

    result_p2 = linprog(
        np.r_[np.zeros(ny), e],
        A_ub=sparse.hstack([A, -E.T], format="csr"),
        b_ub=np.zeros(nx),
        A_eq=sparse.hstack([F, sparse.csr_matrix((nf, ne))], format="csr"),
        b_eq=f,
        bounds=[(0, None)] * ny + [(None, None)] * ne,
        method="highs",
        options=options,
    )
    if not result_p2.success:
        raise RuntimeError(f"Player-two sequence-form LP failed: {result_p2.message}")

    x, q = result_p1.x[:nx], result_p1.x[nx:]
    y, p = result_p2.x[:ny], result_p2.x[ny:]
    value = float(x @ (A @ y))
    lower, upper = float(f @ q), float(e @ p)
    duality_gap = upper - lower
    return {
        "sol_seq_form_p1": x.copy(),
        "sol_seq_form_p2": y.copy(),
        "value": value,
        "lower_bound": lower,
        "upper_bound": upper,
        "gap": max(0.0, duality_gap),
        "duality_gap": duality_gap,
        "feasibility_p1": _max_violation(E, e, x),
        "feasibility_p2": _max_violation(F, f, y),
        "dual_feasibility_p1": max(0.0, float(np.max(F.T @ q - A.T @ x))),
        "dual_feasibility_p2": max(0.0, float(np.max(A @ y - E.T @ p))),
        "optimality_p1": abs(value - lower),
        "optimality_p2": abs(upper - value),
        "iterations_p1": int(result_p1.nit),
        "iterations_p2": int(result_p2.nit),
        "status": "optimal",
    }
