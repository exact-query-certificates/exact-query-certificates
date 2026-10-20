"""Search for exact quantum query algorithms (phase-oracle model of AIN 2017).

State space H_Q (n+1: |0>=no query, |i>=query x_i) x H_W (w) x H_O (2).
Basis index = q*(2w) + wk*2 + o.  Algorithm: U_T O_x ... O_x U_0 |0>, measure o.
O_x = diag((-1)^{x_q}) on H_Q (q=0 unaffected), identity on W, O.
We optimise T+1 unitaries U_t = exp(A_t - A_t^dag) over R random restarts in parallel.
"""
import itertools
import numpy as np
import torch


def all_inputs(n):
    return np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.int64)[:, ::-1]  # row j = bits of j (x_1 = LSB)


def symmetric_fn(n, accept_weights):
    X = all_inputs(n)
    w = X.sum(1)
    return np.isin(w, list(accept_weights)).astype(np.int64), X


class QuerySearch:
    def __init__(self, fvals, X, T, w=1, restarts=32, device="cpu", seed=0):
        self.n = X.shape[1]
        self.T, self.w, self.R = T, w, restarts
        self.D = (self.n + 1) * w * 2
        self.dev = torch.device(device)
        g = torch.Generator().manual_seed(seed)
        dt = torch.complex128
        # oracle phases: (D, M) with M = #inputs
        qidx = np.arange(self.D) // (2 * w)
        sign = np.ones((self.D, len(X)))
        for q in range(1, self.n + 1):
            sign[qidx == q, :] = 1 - 2 * X[:, q - 1][None, :]
        self.sign = torch.tensor(sign, dtype=dt, device=self.dev)
        self.f = torch.tensor(fvals, device=self.dev)
        obit = torch.tensor(np.arange(self.D) % 2, device=self.dev)
        # want[d, x] = 1 if output bit of basis d equals f(x)
        self.want = (obit[:, None] == self.f[None, :]).to(torch.float64)
        scale = 1.0
        A = (torch.randn(restarts, T + 1, self.D, self.D, generator=g, dtype=torch.float64) +
             1j * torch.randn(restarts, T + 1, self.D, self.D, generator=g, dtype=torch.float64)) * scale
        self.A = A.to(dt).to(self.dev).requires_grad_(True)

    def unitaries(self, A=None):
        A = self.A if A is None else A
        return torch.linalg.matrix_exp(A - A.conj().transpose(-1, -2))

    def final_states(self, A=None):
        U = self.unitaries(A)  # (R, T+1, D, D)
        psi = U[:, 0, :, 0]  # U_0 |0>, (R, D)
        psi = psi[:, :, None].expand(-1, -1, self.sign.shape[1])  # (R, D, M)
        for t in range(1, self.T + 1):
            psi = U[:, t] @ (self.sign[None] * psi)
        return psi

    def errors(self, A=None):
        p = self.final_states(A).abs() ** 2  # (R, D, M)
        return 1 - (p * self.want[None]).sum(1)  # (R, M) error prob per input

    def optimize(self, adam_steps=3000, lr=0.05, lbfgs_rounds=40, log=None):
        opt = torch.optim.Adam([self.A], lr=lr)
        for s in range(adam_steps):
            opt.zero_grad()
            e = self.errors()
            loss = (e.mean(1) + e.max(1).values).sum()
            loss.backward()
            opt.step()
            if log and s % 500 == 0:
                log(f"adam {s}: best max-err {e.max(1).values.min().item():.3e}")
        # polish: L-BFGS on sum of errors (sum over restarts is separable)
        opt2 = torch.optim.LBFGS([self.A], lr=1, max_iter=200, history_size=50,
                                 tolerance_grad=1e-30, tolerance_change=1e-30, line_search_fn="strong_wolfe")

        def closure():
            opt2.zero_grad()
            l = self.errors().sum()
            l.backward()
            return l
        for r in range(lbfgs_rounds):
            opt2.step(closure)
            if log and r % 10 == 0:
                with torch.no_grad():
                    log(f"lbfgs {r}: best max-err {self.errors().max(1).values.min().item():.3e}")
        with torch.no_grad():
            e = self.errors()
        return e.max(1).values.cpu().numpy()

    def best(self):
        with torch.no_grad():
            e = self.errors().max(1).values
            r = int(torch.argmin(e))
            return r, float(e[r]), self.unitaries()[r].cpu().numpy()


def verify(unitaries, fvals, X, w):
    """Independent numpy check of a found algorithm (no torch): max error over inputs."""
    n = X.shape[1]
    D = (n + 1) * w * 2
    qidx = np.arange(D) // (2 * w)
    worst = 0.0
    for x, fx in zip(X, fvals):
        ph = np.ones(D)
        for q in range(1, n + 1):
            ph[qidx == q] = (-1) ** x[q - 1]
        psi = unitaries[0][:, 0].copy()
        for U in unitaries[1:]:
            psi = U @ (ph * psi)
        p_ok = np.sum(np.abs(psi[np.arange(D) % 2 == fx]) ** 2)
        worst = max(worst, 1 - p_ok)
    unit_err = max(np.abs(U.conj().T @ U - np.eye(D)).max() for U in unitaries)
    return worst, unit_err


def _skew(theta, D):
    """theta: (D*D,) real -> anti-Hermitian D x D (D^2 real params)."""
    M = theta.reshape(D, D)
    up = torch.triu(M, 1)
    lo = torch.tril(M, -1)
    re = up - up.T  # antisymmetric real part from upper triangle
    im = lo + lo.T + torch.diag(torch.diagonal(M))  # symmetric imaginary part from lower + diag
    return re.to(torch.complex128) + 1j * im.to(torch.complex128)


def polish(unitaries, fvals, X, w, iters=60, tol=1e-28, log=None):
    """Levenberg-Marquardt on residual = amplitudes on wrong output, in local Lie coords.
    Returns (unitaries, history of sum-of-squared-residuals)."""
    n = X.shape[1]
    D = (n + 1) * w * 2
    qidx = np.arange(D) // (2 * w)
    sign = np.ones((D, len(X)))
    for q in range(1, n + 1):
        sign[qidx == q, :] = 1 - 2 * X[:, q - 1][None, :]
    sign = torch.tensor(sign, dtype=torch.complex128)
    wrong = torch.tensor((np.arange(D) % 2)[:, None] != np.asarray(fvals)[None, :])
    Us = [torch.tensor(U, dtype=torch.complex128) for U in unitaries]
    T1 = len(Us)

    def resid(theta, Us):
        th = theta.reshape(T1, D * D)
        V = [torch.linalg.matrix_exp(_skew(th[t], D)) @ Us[t] for t in range(T1)]
        psi = V[0][:, 0][:, None].expand(-1, sign.shape[1])
        for t in range(1, T1):
            psi = V[t] @ (sign * psi)
        r = psi[wrong]
        return torch.cat([r.real, r.imag])

    lam, hist = 1e-3, []
    for it in range(iters):
        z = torch.zeros(T1 * D * D, dtype=torch.float64)
        r = resid(z, Us)
        f0 = float(r @ r)
        hist.append(f0)
        if log:
            log(f"LM {it}: sum err {f0:.3e} lambda {lam:.1e}")
        if f0 < tol:
            break
        J = torch.func.jacfwd(lambda t: resid(t, Us))(z)
        # min-norm damped step: delta = -J^T (J J^T + lam I)^-1 r   (rows << cols)
        JJ = J @ J.T
        while True:
            delta = -J.T @ torch.linalg.solve(JJ + lam * torch.eye(len(r), dtype=torch.float64), r)
            r2 = resid(delta, Us)
            if float(r2 @ r2) < f0:
                th = delta.reshape(T1, D * D)
                Us = [torch.linalg.matrix_exp(_skew(th[t], D)) @ Us[t] for t in range(T1)]
                lam = max(lam / 10, 1e-15)
                break
            lam *= 10
            if lam > 1e8:
                return [U.numpy() for U in Us], hist
    return [U.numpy() for U in Us], hist
