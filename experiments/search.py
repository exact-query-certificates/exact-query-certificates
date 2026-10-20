"""usage: search.py n weights T w restarts lbfgs_rounds seed outprefix [lm_top]
Phase 1: batched L-BFGS over random restarts. Phase 2: LM polish of the best lm_top restarts.
Saves every polished candidate to outprefix_r{rank}.npz with its verified error."""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch
from qkit.query import QuerySearch, symmetric_fn, verify, polish
n, acc, T, w, R, lb, seed, out = (int(sys.argv[1]), {int(v) for v in sys.argv[2].split(",")}, int(sys.argv[3]),
                                  int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), sys.argv[8])
top = int(sys.argv[9]) if len(sys.argv) > 9 else 3
torch.set_num_threads(8)
log = lambda m: print(m, flush=True)
f, X = symmetric_fn(n, acc)
t0 = time.time()
log(f"== n={n} acc={sorted(acc)} T={T} w={w} R={R} seed={seed}")
s = QuerySearch(f, X, T, w=w, restarts=R, seed=seed)
errs = s.optimize(adam_steps=200, lbfgs_rounds=lb, log=lambda m: log("  " + m + f"  [{time.time()-t0:.0f}s]"))
order = np.argsort(errs)
log(f"phase1 sorted max-errs: {np.array2string(errs[order], precision=2)}")
with torch.no_grad():
    allU = s.unitaries().cpu().numpy()
for rank, r in enumerate(order[:top]):
    U2, hist = polish(list(allU[r]), f, X, w, iters=60)
    ve, ue = verify(U2, f, X, w)
    tail = " ".join(f"{h:.1e}" for h in hist[-6:])
    log(f"restart {r} (phase1 {errs[r]:.2e}) -> LM {len(hist)} its, tail [{tail}] -> verified max-err {ve:.2e}, unit {ue:.1e}  [{time.time()-t0:.0f}s]")
    np.savez(f"{out}_r{rank}.npz", U=np.array(U2), f=f, X=X, w=w, T=T, n=n, acc=sorted(acc), err=ve, hist=np.array(hist))
log("done")
