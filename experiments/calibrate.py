"""Calibration: known Q_E values must be found exactly at T=Q_E and fail at T=Q_E-1."""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from qkit.query import QuerySearch, symmetric_fn, verify

# (name, n, accepting weights, known Q_E, source)
CASES = [
    ("PARITY_4", 4, {1, 3}, 2, "folklore ceil(n/2)"),
    ("EXACT_2^4", 4, {2}, 2, "AIS 2013 max(k,n-k)"),
    ("EXACT_{0,4}^4", 4, {0, 4}, 3, "AIN d=4, k=0 (2023): max-1"),
    ("OR_3", 3, {1, 2, 3}, 3, "Q_E(OR_n)=n"),
    ("EXACT_{1,3}^5", 5, {1, 3}, 3, "AIN Thm2 d=2: max(4,3)-1"),
]

w = int(sys.argv[1]) if len(sys.argv) > 1 else 1
for name, n, acc, qe, src in CASES:
    f, X = symmetric_fn(n, acc)
    for T in (qe - 1, qe):
        t0 = time.time()
        s = QuerySearch(f, X, T, w=w, restarts=16, seed=T)
        errs = s.optimize(adam_steps=1500, lbfgs_rounds=20)
        r, e, U = s.best()
        ve, ue = verify(list(U), f, X, w)
        print(f"{name:14s} T={T} (Q_E={qe}, {src}): best max-err {e:.2e} | numpy verify {ve:.2e} "
              f"unitarity {ue:.1e} | restarts<1e-9: {(errs < 1e-9).sum()}/16 | {time.time() - t0:.0f}s", flush=True)
