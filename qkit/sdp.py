"""Barnum-Saks-Szegedy SDP: optimal worst-case error of any T-query algorithm
(phase oracle, unlimited workspace) for f: {0,1}^n -> {0,1}.

Variables: real symmetric PSD M_t^(i) (t=0..T-1, i=0..n), Gamma^0, Gamma^1 (2^n x 2^n).
  sum_i M_0^(i) = J
  sum_i M_t^(i) = sum_i E_i o M_{t-1}^(i)          t = 1..T-1
  Gamma^0 + Gamma^1 = sum_i E_i o M_{T-1}^(i)
  Gamma^{1-f(x)}[x,x] <= eps
E_0 = J, E_i[x,y] = (-1)^(x_i + y_i). Real symmetric suffices (E_i real).
M_t^(i) = Gram matrix of the query-register-|i> components of the states before query t+1.
"""
import numpy as np
import cvxpy as cp


def bss_min_error(fvals, X, T, solver="CLARABEL", verbose=False):
    M, n = X.shape
    J = np.ones((M, M))
    E = [J] + [np.outer(1 - 2 * X[:, i], 1 - 2 * X[:, i]) for i in range(n)]
    eps = cp.Variable()
    cons = []
    prev = None  # sum_i E_i o M_{t-1}^(i) (the Gram matrix of states after the t-th query)
    for t in range(T):
        Ms = [cp.Variable((M, M), PSD=True) for _ in range(n + 1)]
        cons.append(sum(Ms) == (J if prev is None else prev))
        prev = sum(cp.multiply(E[i], Ms[i]) for i in range(n + 1))
    G0, G1 = cp.Variable((M, M), PSD=True), cp.Variable((M, M), PSD=True)
    cons.append(G0 + G1 == (J if prev is None else prev))
    f = np.asarray(fvals)
    cons += [cp.diag(G1)[f == 0] <= eps, cp.diag(G0)[f == 1] <= eps]
    prob = cp.Problem(cp.Minimize(eps), cons)
    prob.solve(solver=solver, verbose=verbose)
    return prob.value, prob
