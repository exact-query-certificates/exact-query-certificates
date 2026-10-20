"""usage: run_one.py n weights(comma) T w restarts [adam] [lbfgs] [seed] [save.npz]"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch
from qkit.query import QuerySearch, symmetric_fn, verify
n, acc, T, w, R = int(sys.argv[1]), {int(v) for v in sys.argv[2].split(",")}, int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
adam = int(sys.argv[6]) if len(sys.argv) > 6 else 2000
lb = int(sys.argv[7]) if len(sys.argv) > 7 else 30
seed = int(sys.argv[8]) if len(sys.argv) > 8 else 0
torch.set_num_threads(8)
f, X = symmetric_fn(n, acc)
t0 = time.time()
s = QuerySearch(f, X, T, w=w, restarts=R, seed=seed)
errs = s.optimize(adam_steps=adam, lbfgs_rounds=lb, log=lambda m: print("  " + m, flush=True))
r, e, U = s.best()
ve, ue = verify(list(U), f, X, w)
print(f"n={n} acc={sorted(acc)} T={T} w={w}: best {e:.2e} verify {ve:.2e} unit {ue:.1e} | sorted errs {np.sort(errs)[:8]} | {time.time()-t0:.0f}s")
if len(sys.argv) > 9:
    np.savez(sys.argv[9], U=U, f=f, X=X, w=w, T=T, n=n, acc=sorted(acc), err=ve)
from qkit.query import polish
U2, hist = polish(list(U), f, X, w, iters=40, log=lambda m: print("  " + m, flush=True))
ve2, ue2 = verify(U2, f, X, w)
print(f"after LM: verify {ve2:.2e} unit {ue2:.1e}")
if len(sys.argv) > 9:
    np.savez(sys.argv[9], U=np.array(U2), f=f, X=X, w=w, T=T, n=n, acc=sorted(acc), err=ve2, hist=hist)
