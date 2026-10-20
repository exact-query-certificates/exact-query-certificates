import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from qkit.query import symmetric_fn
from qkit.sdp_sym import sym_min_error
FULL = {("PARITY_4",1):4.999e-01,("PARITY_4",2):1.4e-07,("EXACT_2^4",1):4.284e-01,("EXACT_2^4",2):2.8e-08,
        ("EXACT_{0,4}^4",2):1.992e-02,("EXACT_{0,4}^4",3):2.0e-08,("OR_3",2):1.996e-02,("OR_3",3):9.8e-09,
        ("EXACT_{1,3}^5",2):4.832e-01,("EXACT_{1,3}^5",3):8.3e-08}
CASES = [("PARITY_4", 4, {1, 3}, 2), ("EXACT_2^4", 4, {2}, 2), ("EXACT_{0,4}^4", 4, {0, 4}, 3),
         ("OR_3", 3, {1, 2, 3}, 3), ("EXACT_{1,3}^5", 5, {1, 3}, 3)]
for name, n, acc, qe in CASES:
    f, X = symmetric_fn(n, acc)
    for T in (qe - 1, qe):
        t0 = time.time()
        v, st, info = sym_min_error(f, n, T)
        print(f"{name:14s} T={T}: sym {v:.3e} vs full {FULL[(name,T)]:.3e} [{st}, {time.time()-t0:.1f}s] orbits S={info['nS']} P={info['nP']} blocks {info['blocksS']}/{info['blocksP']}", flush=True)
