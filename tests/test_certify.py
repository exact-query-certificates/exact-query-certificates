import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, flint
from qkit.certify import exact_psd_full, certify

rng = np.random.default_rng(5)
q = lambda M: [[flint.fmpq(int(a)) for a in row] for row in M]
# PSD rank-deficient integer matrices -> True; perturb one diagonal down -> False
for trial in range(20):
    G = rng.integers(-3, 4, size=(9, 4))
    A = G @ G.T
    assert exact_psd_full(q(A)), "PSD rejected"
    v = rng.integers(-3, 4, size=9)
    while not v.any(): v = rng.integers(-3, 4, size=9)
    # A - c v v^T with c large enough is indefinite unless v in kernel... use explicit indefinite
    B = A.copy(); B[0, 0] -= (A[0, 0] + 1)
    assert not exact_psd_full(q(B)), "negative diagonal accepted"
    C = A + np.outer(v, v)
    assert exact_psd_full(q(C))
    D = np.diag(np.r_[np.ones(8), -1]).astype(int); assert not exact_psd_full(q(D))
    E = A.copy(); E[1, 2] += 1; E[2, 1] += 1  # may or may not be PSD: compare with eigenvalues
    assert exact_psd_full(q(E)) == (np.linalg.eigvalsh(E.astype(float)).min() > -1e-9)
print("exact_psd_full: ok")

for n, acc, T, expect in [(4, {1, 3}, 2, True), (4, {2}, 2, True), (5, {1, 3}, 3, True), (5, {1, 3}, 2, False)]:
    r = certify(n, acc, T, verbose=False)
    print(f"n={n} acc={sorted(acc)} T={T}: certified={r['certified']} (expected {expect}) "
          f"margin={r['numeric']['lam']} {r.get('reason','')}")
    assert r["certified"] == expect
