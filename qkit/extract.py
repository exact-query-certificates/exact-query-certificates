"""Turn a BSS certificate (JSON) into an explicit query algorithm (unitaries) and run it.

States: psi_t(x) = sum_i |i> (x) v_{t,i}(x), with Gram(v_{t,i}) = M_t^(i).
Oracle: phi_t(x) = sum_i (-1)^{x_i} |i> (x) v_{t,i}(x)   (x_0 := 0).
Gram(phi_t) = sum_i E_i o M_t^(i) = Gram(psi_{t+1}), so an isometry maps phi_t(x) -> psi_{t+1}(x)
for every x; we complete it to an orthogonal matrix. Final: phi_{T-1}(x) -> |0>(x)w_0(x) + |1>(x)w_1(x)
with Gram(w_b) = Gamma^b; measuring the first qubit gives f(x) with certainty.
"""
import json
from fractions import Fraction
import numpy as np


def load_full(path):
    C = json.load(open(path))
    n, T = C["n"], C["T"]
    N = 2 ** n
    bits = np.array([[(x >> i) & 1 for i in range(n)] for x in range(N)])
    wt = bits.sum(1)
    def keyS(x, y):
        a, b = sorted((int(wt[x]), int(wt[y])))
        return f"{a},{b},{int((bits[x] & bits[y]).sum())}"
    def keyP(x, y):
        tx, ty = (int(bits[x, 0]), int(wt[x] - bits[x, 0])), (int(bits[y, 0]), int(wt[y] - bits[y, 0]))
        u, v = sorted((tx, ty))
        return f"{u[0]},{u[1]}|{v[0]},{v[1]}|{int((bits[x, 1:] & bits[y, 1:]).sum())}"
    def build(tab, keyf):
        t = {k: float(Fraction(v)) for k, v in tab.items()}
        return np.array([[t.get(keyf(x, y), 0.0) for y in range(N)] for x in range(N)])
    swap = []
    for i in range(n):
        idx = np.arange(N)
        b0, bi = (idx >> 0) & 1, (idx >> i) & 1
        swap.append(np.where(b0 != bi, idx ^ 1 ^ (1 << i), idx))
    Ms = []
    for t in range(T):
        M0, M1 = build(C["M0"][t], keyS), build(C["M1"][t], keyP)
        Ms.append([M0] + [M1[np.ix_(swap[i], swap[i])] for i in range(n)])
    G = [build(C["Gamma0"], keyS), build(C["Gamma1"], keyS)]
    f = np.isin(wt, C["accept"]).astype(int)
    return C, n, T, bits, Ms, G, f


def gram_vectors(M, tol=1e-9):
    """Columns v(x) with v(x).v(y) = M[x,y]; returns (rank, N) array."""
    w, U = np.linalg.eigh((M + M.T) / 2)
    keep = w > tol * max(1.0, w.max())
    return (U[:, keep] * np.sqrt(w[keep])).T


def isometry(A, B, D):
    """Orthogonal D x D matrix U with U A = B (columns), given A^T A == B^T B."""
    # polar factor of B A^T restricted to range(A): U = B A^+ on range(A), complete orthogonally
    Ua, sa, Vta = np.linalg.svd(A, full_matrices=True)
    Ub, sb, Vtb = np.linalg.svd(B, full_matrices=True)
    r = int((sa > 1e-9 * sa[0]).sum())
    # B = W A for orthogonal W: W = Ub_r (Vtb_r Vta_r^T) ... use Procrustes on the r-dim ranges
    M = B @ A.T
    P, s, Qt = np.linalg.svd(M)
    W = P @ Qt  # orthogonal Procrustes solution; exact when Gram(A)=Gram(B)
    return W


def build_algorithm(path):
    C, n, T, bits, Ms, G, f = load_full(path)
    N = 2 ** n
    V = [[gram_vectors(Ms[t][i]) for i in range(n + 1)] for t in range(T)]
    Wb = [gram_vectors(G[b]) for b in (0, 1)]
    r = max([v.shape[0] for vt in V for v in vt] + [w.shape[0] for w in Wb])
    D = max((n + 1) * r, 2 * r)
    sign = 1 - 2 * bits  # (N, n)

    def stack(vt, phase):
        S = np.zeros((D, N))
        for i in range(n + 1):
            vi = vt[i]
            ph = np.ones(N) if (i == 0 or not phase) else sign[:, i - 1]
            S[i * r:i * r + vi.shape[0], :] = vi * ph
        return S

    states_in = [stack(V[t], False) for t in range(T)]      # psi_t
    states_out = [stack(V[t], True) for t in range(T)]      # phi_t = O_x psi_t
    final = np.zeros((D, N))
    final[0:Wb[0].shape[0]] = Wb[0]          # output qubit = block index (0 -> rows [0,r), 1 -> [r,2r))
    final[r:r + Wb[1].shape[0]] = Wb[1]
    Us = [isometry(np.eye(D)[:, :1] @ np.ones((1, N)), states_in[0], D)]  # U_0 |0> = psi_0(x) (x-independent)
    for t in range(1, T):
        Us.append(isometry(states_out[t - 1], states_in[t], D))
    Us.append(isometry(states_out[T - 1], final, D))
    return dict(n=n, T=T, r=r, D=D, Us=Us, f=f, bits=bits, accept=C["accept"])


def run(alg):
    """Simulate the algorithm on every input; return max error probability and unitarity error."""
    n, T, r, D, Us = alg["n"], alg["T"], alg["r"], alg["D"], alg["Us"]
    bits, f = alg["bits"], alg["f"]
    reg = np.arange(D) // r  # query register value of each basis state
    worst = 0.0
    for x in range(2 ** n):
        ph = np.ones(D)
        for i in range(1, n + 1):
            ph[reg == i] = 1 - 2 * bits[x, i - 1]
        psi = Us[0][:, 0].copy()
        for U in Us[1:]:
            psi = U @ (ph * psi)
        out = (np.arange(D) >= r) & (np.arange(D) < 2 * r)   # output bit 1 = second block
        p1 = np.sum(np.abs(psi[out]) ** 2)
        p_wrong = p1 if f[x] == 0 else 1 - p1
        worst = max(worst, p_wrong)
    unit = max(np.abs(U.T @ U - np.eye(D)).max() for U in Us)
    return worst, unit
