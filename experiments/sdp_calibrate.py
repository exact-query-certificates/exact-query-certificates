import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from qkit.query import symmetric_fn
from qkit.sdp import bss_min_error
CASES = [("PARITY_4", 4, {1, 3}, 2), ("EXACT_2^4", 4, {2}, 2), ("EXACT_{0,4}^4", 4, {0, 4}, 3),
         ("OR_3", 3, {1, 2, 3}, 3), ("EXACT_{1,3}^5", 5, {1, 3}, 3)]
for name, n, acc, qe in CASES:
    f, X = symmetric_fn(n, acc)
    for T in (qe - 1, qe):
        t0 = time.time()
        v, p = bss_min_error(f, X, T)
        print(f"{name:14s} T={T} (Q_E={qe}): SDP optimal worst-case error {v:.3e}  [{p.status}, {time.time()-t0:.1f}s]", flush=True)
