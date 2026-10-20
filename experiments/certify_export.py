"""usage: certify_export.py n weights T out.json  -- certify then export JSON certificate"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
from qkit.certify import certify, export_json
n, acc, T, out = int(sys.argv[1]), {int(v) for v in sys.argv[2].split(",")}, int(sys.argv[3]), sys.argv[4]
import os
r = certify(n, acc, T, denom=2 ** int(os.environ.get("CERT_DENOM_BITS", "30")))
if r["certified"]:
    export_json(r["problem"], r["v"], out, meta=dict(
        claim=f"exact {T}-query quantum algorithm for EXACT_{{{min(acc)},{max(acc)}}}^{n}" if len(acc) == 2 else f"exact {T}-query algorithm",
        model="phase oracle O_x|i>=(-1)^{x_i}|i>, O_x|0>=|0>, arbitrary workspace (Ambainis-Iraids-Nagaj 2017)",
        method="Barnum-Saks-Szegedy SDP, S_n-symmetrised, facially reduced, rounded to exact rationals"))
    print("wrote", out)
else:
    print("NOT CERTIFIED:", r.get("reason")); sys.exit(1)
