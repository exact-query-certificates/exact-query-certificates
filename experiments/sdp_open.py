"""usage: sdp_open.py n weights Tmin Tmax"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
from qkit.query import symmetric_fn
from qkit.sdp_sym import sym_min_error
n, acc = int(sys.argv[1]), {int(v) for v in sys.argv[2].split(",")}
k, l = min(acc), max(acc)
conj = max(n - k, l) - 1
f, X = symmetric_fn(n, acc)
for T in range(int(sys.argv[3]), int(sys.argv[4]) + 1):
    t0 = time.time()
    v, st, info = sym_min_error(f, n, T)
    print(f"n={n} acc={sorted(acc)} T={T} (AIN conj/lower bound {conj}, D={n}): SDP min worst-case error {v:.3e} [{st}, {time.time()-t0:.1f}s, blocks {info['blocksS']}/{info['blocksP']}]", flush=True)
