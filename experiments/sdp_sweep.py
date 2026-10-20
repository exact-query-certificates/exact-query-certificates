"""Sweep AIN Conjecture 1 over EXACT_{k,l}^n, d=l-k>=2, canonical rep (k <= n-l), n in [nmin,nmax]."""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
from qkit.query import symmetric_fn
from qkit.sdp_sym import sym_min_error

def status(n, k, l):
    d = l - k
    if d in (2, 3): return "proven(AIN Thm2)"
    if k == 0 or l == n: return "proven(2023 k=0)"
    if k == 1 and l == n - 1: return "proven(2023 1,n-1)"
    return "OPEN"

nmin, nmax = int(sys.argv[1]), int(sys.argv[2])
dmin = int(sys.argv[3]) if len(sys.argv) > 3 else 4
for n in range(nmin, nmax + 1):
    for k in range(0, n + 1):
        for l in range(k + dmin, n + 1):
            if k > n - l: continue  # canonical under x -> complement
            conj = max(n - k, l) - 1
            f, X = symmetric_fn(n, {k, l})
            t0 = time.time()
            lo, st1, _ = sym_min_error(f, n, conj - 1)
            at, st2, _ = sym_min_error(f, n, conj)
            print(f"n={n:2d} k={k} l={l:2d} d={l-k} {status(n,k,l):20s} conj={conj}: err(T=conj-1)={lo:.2e} err(T=conj)={at:.2e} [{st1[:7]},{st2[:7]} {time.time()-t0:.0f}s]", flush=True)
