"""Symmetry-reduced BSS SDP for SYMMETRIC f (f depends only on |x|).

Averaging a feasible solution over S_n (acting on input coordinates and on the query
index i>=1) keeps it feasible and keeps the max error, so WLOG:
  M_t^(0), Gamma^0, Gamma^1   S_n-invariant      (orbit of (x,y) = |x|,|y|,|x&y|)
  M_t^(i) = P_i M_t^(1) P_i^T, M_t^(1) S_{n-1}-invariant (fixes coordinate 0)
Invariant matrices are parametrised by orbit coefficients; PSD-ness is checked on
the blocks Q_k^T A Q_k (one copy of each isotypic component, Schur's lemma).
Equalities between S_n-invariant matrices are imposed at one entry per orbit.
"""
import itertools
import numpy as np
import cvxpy as cp


def inputs(n):
    return np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.int64)[:, ::-1]


def _lowest_weight_vector(Xs, k, rng):
    """A nonzero vector on weight-k strings of Xs annihilated by the lowering operator."""
    wts = Xs.sum(1)
    Wk = np.where(wts == k)[0]
    if k == 0:
        v = np.zeros(len(Xs)); v[Wk] = 1
        return v
    Wk1 = np.where(wts == k - 1)[0]
    pos = {tuple(Xs[i]): r for r, i in enumerate(Wk1)}
    R = np.zeros((len(Wk1), len(Wk)))
    for c, i in enumerate(Wk):
        for b in np.where(Xs[i] == 1)[0]:
            y = Xs[i].copy(); y[b] = 0
            R[pos[tuple(y)], c] = 1
    _, s, Vt = np.linalg.svd(R)
    null = Vt[(s > 1e-9).sum():]
    v = np.zeros(len(Xs))
    v[Wk] = rng.normal(size=len(null)) @ null
    return v


def _raise(Xs, v):
    """L v: L|x> = sum over 0->1 flips."""
    idx = {tuple(x): r for r, x in enumerate(Xs)}
    out = np.zeros_like(v)
    for r in np.nonzero(np.abs(v) > 0)[0]:
        for b in np.where(Xs[r] == 0)[0]:
            y = Xs[r].copy(); y[b] = 1
            out[idx[tuple(y)]] += v[r]
    return out


def sym_bases(m, rng):
    """For S_m on {0,1}^m: list over k of Q_k (2^m x (m-2k+1)), orthonormal columns."""
    Xs = inputs(m)
    Qs = []
    for k in range(m // 2 + 1):
        v = _lowest_weight_vector(Xs, k, rng)
        cols = []
        for j in range(k, m - k + 1):
            cols.append(v / np.linalg.norm(v))
            v = _raise(Xs, v)
        Qs.append(np.array(cols).T)
    return Qs


def build(fvals, n, T, exact=False, seed=0):
    rng = np.random.default_rng(seed)
    X = inputs(n)
    Mn = len(X)
    wt = X.sum(1)
    inter = X @ X.T
    # S_n orbits on (unordered) pairs: key (min(|x|,|y|), max, |x&y|)
    a, b = np.minimum.outer(wt, wt), np.maximum.outer(wt, wt)
    keyS = a * (n + 1) ** 2 + b * (n + 1) + inter
    uS, orbS = np.unique(keyS, return_inverse=True)
    orbS = orbS.reshape(Mn, Mn)
    # S_{n-1} orbits (coordinate 0 fixed): key from ((x0,|x'|),(y0,|y'|)) unordered + |x'&y'|
    x0, wp = X[:, 0], X[:, 1:].sum(1)
    tag = x0 * (n + 1) + wp
    ta, tb = np.minimum.outer(tag, tag), np.maximum.outer(tag, tag)
    interp = X[:, 1:] @ X[:, 1:].T
    keyP = (ta * (2 * n + 2) + tb) * (n + 1) + interp
    uP, orbP = np.unique(keyP, return_inverse=True)
    orbP = orbP.reshape(Mn, Mn)
    nS, nP = len(uS), len(uP)

    # representative entry for every S_n orbit
    reps = [tuple(np.argwhere(orbS == o)[0]) for o in range(nS)]
    # coordinate swap (0 <-> i) as an index permutation of inputs
    perms = []
    for i in range(n):
        Y = X.copy(); Y[:, [0, i]] = Y[:, [i, 0]]
        perms.append(Y @ (1 << np.arange(n)))
    sgn = 1 - 2 * X  # (Mn, n)

    # linear maps evaluated at reps:  sum_i M^(i)[x,y]  and  sum_i E_i o M^(i)[x,y]
    SumS = np.zeros((nS, nS)); SumP = np.zeros((nS, nP))
    PhS = np.zeros((nS, nS)); PhP = np.zeros((nS, nP))
    for r, (x, y) in enumerate(reps):
        SumS[r, orbS[x, y]] += 1
        PhS[r, orbS[x, y]] += 1
        for i in range(n):
            o = orbP[perms[i][x], perms[i][y]]
            SumP[r, o] += 1
            PhP[r, o] += sgn[x, i] * sgn[y, i]

    # PSD block images
    QS = sym_bases(n, rng)
    Qm = sym_bases(n - 1, rng)
    QP = []
    for Q in Qm:  # extend with coordinate-0 bit (low bit of index): index = 2*x' + x0
        Z = np.zeros((Mn, 2 * Q.shape[1]))
        for b0 in (0, 1):
            Z[b0::2, b0 * Q.shape[1]:(b0 + 1) * Q.shape[1]] = Q
        QP.append(Z)

    def block_images(orb, nO, Qs):
        imgs = []
        for Q in Qs:
            imgs.append(np.array([Q.T @ (orb == o).astype(float) @ Q for o in range(nO)]))
        return imgs
    BS, BP = block_images(orbS, nS, QS), block_images(orbP, nP, QP)

    def psd(c, imgs):
        cons = []
        for I in imgs:
            d = I.shape[1]
            B = cp.reshape(I.reshape(I.shape[0], d * d).T @ c, (d, d), order="C")
            cons.append(0.5 * (B + B.T) >> 0)
        return cons

    J = np.ones(nS)  # all-ones matrix in S_n-orbit coords
    # At an S_n-orbit rep, M^(0) contributes its own orbit coefficient to both maps,
    # so SumS = PhS = I and the equalities eliminate m0 and g0 (better conditioning).
    assert np.allclose(SumS, np.eye(nS)) and np.allclose(PhS, np.eye(nS))
    cons, prev, m1s = [], J, []
    for t in range(T):
        m1 = cp.Variable(nP)
        m0 = prev - SumP @ m1
        cons += psd(m0, BS) + psd(m1, BP)
        prev = m0 + PhP @ m1
        m1s.append(m1)
    g1 = cp.Variable(nS)
    g0 = prev - g1
    cons += psd(g0, BS) + psd(g1, BS)
    eps = cp.Variable()
    f = np.asarray(fvals)
    diag_orb = {w: orbS[int(np.where(wt == w)[0][0]), int(np.where(wt == w)[0][0])] for w in range(n + 1)}
    fw = {w: int(f[np.where(wt == w)[0][0]]) for w in range(n + 1)}
    for w in range(n + 1):
        bad = g1 if fw[w] == 0 else g0
        cons.append(bad[diag_orb[w]] == 0 if exact else bad[diag_orb[w]] <= eps)
    prob = cp.Problem(cp.Minimize(0 if exact else eps), cons)
    info = dict(nS=nS, nP=nP, blocksS=[Q.shape[1] for Q in QS], blocksP=[Q.shape[1] for Q in QP])
    return prob, info


def sym_min_error(fvals, n, T, solver="CLARABEL", **kw):
    prob, info = build(fvals, n, T, **kw)
    prob.solve(solver=solver)
    return prob.value, prob.status, info
