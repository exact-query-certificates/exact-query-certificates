"""Exact rational certificate that an exact T-query algorithm exists for a symmetric f.

By Barnum-Saks-Szegedy, a T-query algorithm with error 0 exists iff there are PSD
M_t^(i) (t<T, i=0..n), Gamma^0, Gamma^1 with
    sum_i M_0^(i) = J,  sum_i M_t^(i) = sum_i E_i o M_{t-1}^(i),
    Gamma^0 + Gamma^1 = sum_i E_i o M_{T-1}^(i),  Gamma^b[x,x] = 0 when f(x) != b.
Pipeline: S_n-reduced variables (see sdp_sym), integer Specht bases, facial reduction
(after t queries amplitudes have degree <= t; Gamma^b lives on f^{-1}(b)), numeric
max-margin solve, exact RREF rounding, exact positive-definiteness, and finally an
independent check of the raw BSS conditions on full 2^n x 2^n rational matrices.
"""
import itertools
import time
from fractions import Fraction
import numpy as np
import cvxpy as cp
import flint


def inputs(n):
    return np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.int64)[:, ::-1]


def specht_vector(m, k):
    """Integer lowest-weight vector of the (m-k,k) isotypic part on weight-k strings:
    pairs (0,1),(2,3),..; x has exactly one 1 per pair, 0 elsewhere; sign = prod(+1 if
    first of pair is 1). Annihilated by the lowering operator."""
    X = inputs(m)
    v = np.zeros(len(X), dtype=object)
    for r, x in enumerate(X):
        if x.sum() != k or x[2 * k:].sum() != 0:
            continue
        if all(x[2 * j] + x[2 * j + 1] == 1 for j in range(k)):
            s = 1
            for j in range(k):
                s *= 1 if x[2 * j] else -1
            v[r] = s
    return X, v


def raise_op(X, v):
    idx = {tuple(x): r for r, x in enumerate(X)}
    out = np.zeros(len(v), dtype=object)
    for r in np.nonzero(v)[0]:
        for b in np.where(X[r] == 0)[0]:
            y = X[r].copy(); y[b] = 1
            out[idx[tuple(y)]] += v[r]
    return out


def int_bases(m):
    out = []
    for k in range(m // 2 + 1):
        X, v = specht_vector(m, k)
        cols = []
        for _ in range(k, m - k + 1):
            g = 0
            for a in v:
                g = np.gcd(g, int(a))
            cols.append(v // g)  # same direction, smaller integers (better conditioning)
            v = raise_op(X, v)
        assert not np.any(v), "raising past top weight should vanish"
        out.append(np.array(cols, dtype=object).T)
    return out


def characters(n, maxdeg=None, mindeg=0):
    X = inputs(n)
    rows = []
    for S in range(2 ** n):
        d = bin(S).count("1")
        if mindeg <= d <= (n if maxdeg is None else maxdeg):
            sbits = np.array([(S >> i) & 1 for i in range(n)])
            rows.append(1 - 2 * ((X @ sbits) % 2))
    return np.array(rows, dtype=object).reshape(-1, len(X))


def int_nullspace(A):
    """Integer basis (columns) of {c : A c = 0} for an integer matrix A (object array)."""
    A = np.asarray(A, dtype=object)
    if A.shape[0] == 0:
        return np.eye(A.shape[1], dtype=object)
    N, nul = flint.fmpz_mat([[int(a) for a in row] for row in A]).nullspace()
    cols = [[int(N[i, j]) for i in range(N.nrows())] for j in range(nul)]
    return np.array(cols, dtype=object).T.reshape(A.shape[1], nul)


def col_basis(A):
    """Integer basis (columns) of the column space of an integer matrix."""
    A = np.asarray(A, dtype=object)
    if A.shape[1] == 0:
        return A
    R, r = flint.fmpq_mat([[int(a) for a in row] for row in A.T]).rref()
    rows = []
    for i in range(r):
        den = 1
        for j in range(R.ncols()):
            den = den * R[i, j].q // np.gcd(den, R[i, j].q) if R[i, j] != 0 else den
        rows.append([int(R[i, j] * den) for j in range(R.ncols())])
    return np.array(rows, dtype=object).T.reshape(A.shape[0], r)


def intersect(U, V):
    """Integer basis of span(U) intersect span(V)."""
    if U.shape[1] == 0 or V.shape[1] == 0:
        return np.zeros((U.shape[0], 0), dtype=object)
    N = int_nullspace(np.hstack([U, -V]))
    if N.shape[1] == 0:
        return np.zeros((U.shape[0], 0), dtype=object)
    return col_basis(U @ N[:U.shape[1]])


class Problem:
    def __init__(self, n, accept, T):
        self.n, self.T = n, T
        X = inputs(n)
        self.X = X
        Mn = len(X)
        self.f = np.isin(X.sum(1), list(accept)).astype(np.int64)
        wt, inter = X.sum(1), X @ X.T
        a, b = np.minimum.outer(wt, wt), np.maximum.outer(wt, wt)
        _, orbS = np.unique(a * (n + 1) ** 2 + b * (n + 1) + inter, return_inverse=True)
        self.orbS = orbS.reshape(Mn, Mn)
        tag = X[:, 0] * (n + 1) + X[:, 1:].sum(1)
        ta, tb = np.minimum.outer(tag, tag), np.maximum.outer(tag, tag)
        _, orbP = np.unique((ta * (2 * n + 2) + tb) * (n + 1) + X[:, 1:] @ X[:, 1:].T, return_inverse=True)
        self.orbP = orbP.reshape(Mn, Mn)
        self.nS, self.nP = self.orbS.max() + 1, self.orbP.max() + 1
        reps = [tuple(np.argwhere(self.orbS == o)[0]) for o in range(self.nS)]
        self.perms = []
        for i in range(n):
            Y = X.copy(); Y[:, [0, i]] = Y[:, [i, 0]]
            self.perms.append(Y @ (1 << np.arange(n)))
        sgn = 1 - 2 * X
        self.SumP = np.zeros((self.nS, self.nP), dtype=object)
        self.PhP = np.zeros((self.nS, self.nP), dtype=object)
        for r, (x, y) in enumerate(reps):
            assert self.orbS[x, y] == r
            for i in range(n):
                o = self.orbP[self.perms[i][x], self.perms[i][y]]
                self.SumP[r, o] += 1
                self.PhP[r, o] += int(sgn[x, i] * sgn[y, i])
        # integer block bases
        self.QS = int_bases(n)
        self.QP = []
        for Q in int_bases(n - 1):  # index = 2*x' + x0  (x0 = coordinate 0 = low bit)
            Z = np.zeros((Mn, 2 * Q.shape[1]), dtype=object)
            for b0 in (0, 1):
                Z[b0::2, b0 * Q.shape[1]:(b0 + 1) * Q.shape[1]] = Q
            self.QP.append(Z)
        self.IS = [self._images(self.orbS, self.nS, Q) for Q in self.QS]
        self.IP = [self._images(self.orbP, self.nP, Q) for Q in self.QP]
        # ---- faces: subspaces of R^{2^n} that must contain the range of each Gram matrix ----
        # Final amplitude functions on output b lie in W_b = {h in V_T : h = 0 off f^{-1}(b)}.
        # A fixed unitary preserves the span of amplitude functions and the oracle multiplies
        # register i by chi_i, so register-i components before query t+1 lie in
        # F_t^(i) = chi_i * Z_{t+1}  intersect  V_t,  and Z_t = sum_i F_t^(i)  (Z_T = W_0 + W_1).
        self.Hle = [characters(n, maxdeg=t) for t in range(T + 1)]
        V = [col_basis(self.Hle[t].T) for t in range(T + 1)]
        Ht = self.Hle[T].T
        self.W = []
        for bit in (0, 1):
            alpha = int_nullspace(Ht[self.f != bit])
            self.W.append(col_basis(Ht @ alpha) if alpha.shape[1] else np.zeros((Mn, 0), dtype=object))
        chi0 = (1 - 2 * X[:, 0]).astype(object)
        Zs = col_basis(np.hstack(self.W))
        self.face0, self.face1 = [None] * T, [None] * T
        for t in reversed(range(T)):
            self.face0[t] = intersect(Zs, V[t])
            self.face1[t] = intersect(chi0[:, None] * Zs, V[t])
            parts = [self.face0[t]] + [self.face1[t][self.perms[i]] for i in range(n)]
            Zs = col_basis(np.hstack(parts))
        kern = lambda Fb, Q: int_nullspace(Fb.T @ Q) if Fb.shape[1] else np.eye(Q.shape[1], dtype=object)
        self.GS = [[kern(self.face0[t], Q) for Q in self.QS] for t in range(T)]
        self.GP = [[kern(self.face1[t], Q) for Q in self.QP] for t in range(T)]
        self.GG = [[kern(self.W[bit], Q) for Q in self.QS] for bit in (0, 1)]
        self.face_dims = dict(Z0=Zs.shape[1], M0=[F.shape[1] for F in self.face0],
                              M1=[F.shape[1] for F in self.face1], W=[w.shape[1] for w in self.W])
        self.NV = T * self.nP + self.nS

    @staticmethod
    def _images(orb, nO, Q):
        Qf = Q.astype(object)
        imgs = np.zeros((nO, Q.shape[1], Q.shape[1]), dtype=object)
        for o in range(nO):
            A = (orb == o).astype(np.int64).astype(object)
            imgs[o] = Qf.T @ A @ Qf
        return imgs

    # affine expressions in v as (A (rows x NV), b (rows,)) with object/int entries
    def affine_terms(self):
        nS, nP, T, NV = self.nS, self.nP, self.T, self.NV
        Z = lambda r: np.zeros((r, NV), dtype=object)
        prev = (Z(nS), np.ones(nS, dtype=object))
        terms = []
        for t in range(T):
            m1A = Z(nP); m1A[:, t * nP:(t + 1) * nP] = np.eye(nP, dtype=np.int64).astype(object)
            m1 = (m1A, np.zeros(nP, dtype=object))
            m0 = (prev[0] - self.SumP @ m1A, prev[1].copy())
            terms.append(("M0", t, m0, self.IS, self.GS[t], self.QS))
            terms.append(("M1", t, m1, self.IP, self.GP[t], self.QP))
            prev = (m0[0] + self.PhP @ m1A, m0[1].copy())
        g1A = Z(nS); g1A[:, T * nP:] = np.eye(nS, dtype=np.int64).astype(object)
        g1 = (g1A, np.zeros(nS, dtype=object))
        g0 = (prev[0] - g1A, prev[1].copy())
        terms.append(("G0", T, g0, self.IS, self.GG[0], self.QS))
        terms.append(("G1", T, g1, self.IS, self.GG[1], self.QS))
        return terms

    @staticmethod
    def block_affine(aff, I):
        """Block as affine map: flat (d*d) rows."""
        A, b = aff
        nO, d, _ = I.shape
        If = I.reshape(nO, d * d).T  # (d*d, nO)
        return If @ A, If @ b, d

    def equalities(self):
        rows, rhs = [], []
        for name, t, aff, imgs, Gs, _ in self.affine_terms():
            for I, G in zip(imgs, Gs):
                BA, Bb, d = self.block_affine(aff, I)
                if G.shape[1] == 0:
                    continue
                BA3, Bb2 = BA.reshape(d, d, -1), Bb.reshape(d, d)
                for g in G.T:
                    rA = np.einsum("ijv,j->iv", BA3, g)  # (B g)_i
                    rb = Bb2 @ g
                    for i in range(d):
                        rows.append(rA[i]); rhs.append(-rb[i])
        return np.array(rows, dtype=object), np.array(rhs, dtype=object)

    def faces(self):
        """For each PSD block: (name, t, block-affine, F integer basis of face complement)."""
        out = []
        for name, t, aff, imgs, Gs, Qs in self.affine_terms():
            for k, (I, G, Q) in enumerate(zip(imgs, Gs, Qs)):
                BA, Bb, d = self.block_affine(aff, I)
                F = int_nullspace(G.T) if G.shape[1] else np.eye(d, dtype=object)
                out.append((f"{name}_{t}_k{k}", BA, Bb, d, F, Q))
        return out


def exact_param(P):
    """Exact RREF of the equality system: v = v_part + N z over the free variables.
    Returns None if inconsistent. Cached on P."""
    if getattr(P, "_param", None) is not None:
        return P._param
    Aeq, beq = P.equalities()
    M = flint.fmpq_mat([[int(x) for x in row] + [int(bb)] for row, bb in zip(Aeq, beq)])
    R, rank = M.rref()
    NV = P.NV
    pivots, rowpiv = [], []
    for i in range(rank):
        for j in range(NV + 1):
            if R[i, j] != 0:
                if j == NV:
                    P._param = dict(consistent=False, rank=rank, rows=len(beq))
                    return P._param
                pivots.append(j); rowpiv.append(i)
                break
    free = [j for j in range(NV) if j not in set(pivots)]
    P._param = dict(consistent=True, R=R, rank=rank, rows=len(beq), pivots=pivots, rowpiv=rowpiv, free=free)
    return P._param


def numeric_solve(P, verbose=False):
    par = exact_param(P)
    if not par["consistent"]:
        return dict(v=None, lam=None, status="inconsistent", eq_rank=par["rank"], eq_rows=par["rows"],
                    particular_residual=np.inf, free_dims=0)
    R, NV, free = par["R"], P.NV, par["free"]
    v0 = np.zeros(NV); N = np.zeros((NV, len(free)))
    for c, j in enumerate(free):
        N[j, c] = 1.0
    for i, p in zip(par["rowpiv"], par["pivots"]):
        v0[p] = float(R[i, NV])
        for c, j in enumerate(free):
            if R[i, j] != 0:
                N[p, c] = -float(R[i, j])
    r, res = par["rank"], 0.0
    # same affine set, better conditioned: orthonormal null-space basis, min-norm particular point
    N = np.linalg.qr(N)[0] if N.shape[1] else N
    v0 = v0 - N @ (N.T @ v0)
    z, lam = cp.Variable(N.shape[1]), cp.Variable()
    cons = []
    for name, BA, Bb, d, F, Q in P.faces():
        if F.shape[1] == 0:
            continue
        # normalised coordinates: Bn = D^-1 B D^-1 with D = column norms of the integer basis Q
        Dn = np.sqrt((Q.astype(float) ** 2).sum(0))
        # B c = 0  <=>  Bn (D c) = 0, so the normalised face is D^-1 span(F)
        Fo = np.linalg.qr(F.astype(float) / Dn[:, None])[0]
        Fo = Fo / Dn[:, None]  # Fo^T B Fo == Fo_n^T Bn Fo_n
        BAf, Bbf = BA.astype(float), Bb.astype(float)
        Bexpr = cp.reshape(BAf @ N @ z + BAf @ v0 + Bbf, (d, d), order="C")
        R = Fo.T @ (0.5 * (Bexpr + Bexpr.T)) @ Fo
        cons.append(R - lam * np.eye(Fo.shape[1]) >> 0)
    cons.append(lam <= 1)
    prob = cp.Problem(cp.Maximize(lam), cons)
    try:
        prob.solve(solver="CLARABEL", verbose=verbose)
    except cp.error.SolverError:
        prob.solve(solver="SCS", eps=1e-10, max_iters=200000, verbose=verbose)
    v = v0 + N @ z.value
    return dict(v=v, lam=lam.value, status=prob.status, eq_rank=r, eq_rows=par["rows"], particular_residual=res,
                free_dims=N.shape[1])


def to_fmpq(a):
    return flint.fmpq(a.numerator, a.denominator) if isinstance(a, Fraction) else flint.fmpq(int(a))


def exact_round(P, v_num, denom=2 ** 30):
    par = exact_param(P)
    assert par["consistent"]
    R, rank, NV = par["R"], par["rank"], P.NV
    pivots, rowpiv, free = par["pivots"], par["rowpiv"], par["free"]
    v = [None] * NV
    for j in free:
        v[j] = flint.fmpq(int(round(v_num[j] * denom)), denom)
    for i, p in zip(rowpiv, pivots):
        acc = R[i, NV]
        for j in free:
            if R[i, j] != 0:
                acc -= R[i, j] * v[j]
        v[p] = acc
    return v, dict(rank=rank, free=len(free))


def exact_pd(B):
    """Sylvester: all leading principal minors > 0 (exact)."""
    d = B.nrows()
    for k in range(1, d + 1):
        sub = flint.fmpq_mat([[B[i, j] for j in range(k)] for i in range(k)])
        if sub.det() <= 0:
            return False, k
    return True, d


def exact_check_blocks(P, v):
    vq = flint.fmpq_mat([[x] for x in v])
    report = []
    ok_all = True
    for name, BA, Bb, d, F, _ in P.faces():
        BAq = flint.fmpq_mat([[int(x) for x in row] for row in BA])
        Bflat = BAq * vq
        B = flint.fmpq_mat([[Bflat[i * d + j, 0] + int(Bb[i * d + j]) for j in range(d)] for i in range(d)])
        sym = all(B[i, j] == B[j, i] for i in range(d) for j in range(d))
        Fq = flint.fmpq_mat([[int(x) for x in row] for row in F]) if F.shape[1] else None
        if Fq is None:
            zero = all(B[i, j] == 0 for i in range(d) for j in range(d))
            report.append((name, d, 0, sym and zero)); ok_all &= sym and zero
            continue
        Red = Fq.transpose() * B * Fq
        pd, _ = exact_pd(Red)
        # B G = 0 is enforced by the equalities; re-check it exactly here
        G = int_nullspace(np.array([[int(x) for x in row] for row in F.T], dtype=object))
        kerok = True
        if G.shape[1]:
            Gq = flint.fmpq_mat([[int(x) for x in row] for row in G])
            BG = B * Gq
            kerok = all(BG[i, j] == 0 for i in range(BG.nrows()) for j in range(BG.ncols()))
        report.append((name, d, F.shape[1], sym and pd and kerok))
        ok_all &= sym and pd and kerok
    return ok_all, report


# ---------------- independent check on full 2^n x 2^n matrices ----------------------
def exact_psd_full(A):
    """Exact PSD test by symmetric elimination: a zero pivot requires a zero row."""
    A = [row[:] for row in A]
    d = len(A)
    if any(A[i][j] != A[j][i] for i in range(d) for j in range(i + 1, d)):
        return False
    for k in range(d):
        p = A[k][k]
        if p < 0:
            return False
        rowk = A[k]
        if p == 0:
            if any(rowk[j] != 0 for j in range(k + 1, d)):
                return False
            continue
        nz = [j for j in range(k + 1, d) if rowk[j] != 0]
        for i in nz:
            fct = rowk[i] / p
            Ai = A[i]
            for j in nz:
                Ai[j] -= fct * rowk[j]
    return True


def full_check(P, v):
    """Rebuild every Gram matrix exactly and test the raw BSS conditions."""
    n, T, nS, nP = P.n, P.T, P.nS, P.nP
    Mn = 2 ** n
    X = P.X
    sgn = [[1 - 2 * int(X[x, i]) for i in range(n)] for x in range(Mn)]
    vals = lambda A, b: [sum((A[r, j] * v[j] for j in np.nonzero(A[r])[0]), flint.fmpq(int(b[r]))) for r in range(A.shape[0])]
    terms = {(nm, t): aff for nm, t, aff, _, _, _ in P.affine_terms()}
    orbS, orbP = P.orbS, P.orbP
    full = lambda coef, orb: [[coef[orb[x, y]] for y in range(Mn)] for x in range(Mn)]
    ok = True
    log = []
    J = [[flint.fmpq(1)] * Mn for _ in range(Mn)]
    prev = J
    for t in range(T):
        c0 = vals(*terms[("M0", t)]); c1 = vals(*terms[("M1", t)])
        M0 = full(c0, orbS)
        M1 = full(c1, orbP)
        Ms = [M0] + [[[M1[P.perms[i][x]][P.perms[i][y]] for y in range(Mn)] for x in range(Mn)] for i in range(n)]
        # sum_i M^(i) == prev
        s_ok = all(sum((Ms[i][x][y] for i in range(n + 1)), flint.fmpq(0)) == prev[x][y]
                   for x in range(Mn) for y in range(Mn))
        p_ok = exact_psd_full(M0) and exact_psd_full(M1)  # M^(i) are permutations of M^(1)
        log.append(f"t={t}: sum_i M^(i) == previous Gram: {s_ok}; M^(0), M^(1) PSD: {p_ok}")
        ok &= s_ok and p_ok
        prev = [[Ms[0][x][y] + sum((sgn[x][i - 1] * sgn[y][i - 1] * Ms[i][x][y] for i in range(1, n + 1)), flint.fmpq(0))
                 for y in range(Mn)] for x in range(Mn)]
    g0 = full(vals(*terms[("G0", T)]), orbS)
    g1 = full(vals(*terms[("G1", T)]), orbS)
    s_ok = all(g0[x][y] + g1[x][y] == prev[x][y] for x in range(Mn) for y in range(Mn))
    p_ok = exact_psd_full(g0) and exact_psd_full(g1)
    f = P.f
    e_ok = all((g1[x][x] == 0) if f[x] == 0 else (g0[x][x] == 0) for x in range(Mn))
    log.append(f"final: Gamma0+Gamma1 == final Gram: {s_ok}; PSD: {p_ok}; zero error on every input: {e_ok}")
    ok &= s_ok and p_ok and e_ok
    return ok, log


def certify(n, accept, T, denom=2 ** 30, verbose=True):
    say = print if verbose else (lambda *a, **k: None)
    t0 = time.time()
    P = Problem(n, accept, T)
    say(f"[{time.time()-t0:.0f}s] built: nS={P.nS} nP={P.nP} NV={P.NV}")
    sol = numeric_solve(P)
    if sol["status"] == "inconsistent":
        say(f"[{time.time()-t0:.0f}s] exact RREF: equalities INCONSISTENT (rank {sol['eq_rank']}/{sol['eq_rows']})")
        return dict(certified=False, reason="face/exactness equalities inconsistent (no exact algorithm)", numeric=sol)
    say(f"[{time.time()-t0:.0f}s] numeric: status={sol['status']} margin lambda={sol['lam']:.3e} "
        f"eq rank {sol['eq_rank']}/{sol['eq_rows']} free dims {sol['free_dims']}")
    if sol["lam"] is None or sol["lam"] <= 1e-7:
        return dict(certified=False, reason="no strictly feasible point on the predicted face", numeric=sol)
    v, info = exact_round(P, sol["v"], denom)
    say(f"[{time.time()-t0:.0f}s] exact rref rank {info['rank']}, free {info['free']}")
    ok_b, rep = exact_check_blocks(P, v)
    say(f"[{time.time()-t0:.0f}s] reduced blocks exactly PD (+ kernel equalities): {ok_b}")
    if not ok_b:
        for r in rep:
            if not r[3]:
                say("   FAIL", r)
        return dict(certified=False, reason="exact block check failed", report=rep, numeric=sol)
    ok_f, log = full_check(P, v)
    for line in log:
        say("   " + line)
    say(f"[{time.time()-t0:.0f}s] independent full-matrix BSS check: {ok_f}")
    return dict(certified=ok_b and ok_f, v=v, numeric=sol, report=rep, full_log=log, problem=P)


def export_json(P, v, path, meta=None):
    """Write the certificate with human-readable orbit keys (see certificates/verify_certificate.py)."""
    import json
    X, n = P.X, P.n
    wt = X.sum(1)
    def keyS(x, y):
        a, b = sorted((int(wt[x]), int(wt[y])))
        return f"{a},{b},{int((X[x] & X[y]).sum())}"
    def keyP(x, y):
        tx = (int(X[x, 0]), int(wt[x] - X[x, 0])); ty = (int(X[y, 0]), int(wt[y] - X[y, 0]))
        u, w = sorted((tx, ty))
        return f"{u[0]},{u[1]}|{w[0]},{w[1]}|{int((X[x, 1:] & X[y, 1:]).sum())}"
    repS = [tuple(np.argwhere(P.orbS == o)[0]) for o in range(P.nS)]
    repP = [tuple(np.argwhere(P.orbP == o)[0]) for o in range(P.nP)]
    terms = {(nm, t): aff for nm, t, aff, _, _, _ in P.affine_terms()}
    def coeffs(aff):
        A, b = aff
        return [sum((A[r, j] * v[j] for j in np.nonzero(A[r])[0]), flint.fmpq(int(b[r]))) for r in range(A.shape[0])]
    def table(c, reps, keyf):
        out = {}
        for o, (x, y) in enumerate(reps):
            k = keyf(x, y)
            assert k not in out
            out[k] = str(c[o])
        return out
    doc = dict(meta or {}, n=n, T=P.T, accept=sorted(int(w) for w in set(wt[P.f == 1])),
               M0=[table(coeffs(terms[("M0", t)]), repS, keyS) for t in range(P.T)],
               M1=[table(coeffs(terms[("M1", t)]), repP, keyP) for t in range(P.T)],
               Gamma0=table(coeffs(terms[("G0", P.T)]), repS, keyS),
               Gamma1=table(coeffs(terms[("G1", P.T)]), repS, keyS))
    json.dump(doc, open(path, "w"), indent=1)
    return doc
