"""usage: certify_run.py n weights T out.pkl"""
import sys, pathlib, pickle
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
from qkit.certify import certify
n, acc, T = int(sys.argv[1]), {int(v) for v in sys.argv[2].split(",")}, int(sys.argv[3])
r = certify(n, acc, T)
print("face dims:", r["problem"].face_dims if "problem" in r else None)
print("CERTIFIED" if r["certified"] else f"NOT CERTIFIED: {r.get('reason')}")
if r["certified"]:
    pickle.dump(dict(n=n, acc=sorted(acc), T=T, v=[(int(x.p), int(x.q)) for x in r["v"]],
                     full_log=r["full_log"]), open(sys.argv[4], "wb"))
