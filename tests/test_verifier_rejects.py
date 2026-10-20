"""Shows that the checkers are not rubber stamps: from the valid n = 7 certificate, build ~30
subtly broken ones and confirm that BOTH checkers reject every one (and accept the original).

    python tests/test_verifier_rejects.py          (about a minute; needs python-flint)

Breakages: wrong function (accept set changed), one query fewer (drop a layer), Gamma0/Gamma1
swapped, sum-preserving 1e-12 shifts between Gamma0 and Gamma1, and 1e-9 perturbations of M1 that
are compensated so that EVERY linear condition still holds (only the PSD test can catch those).
"""
import copy
import json
import pathlib
import random
import subprocess
import sys
import tempfile
from fractions import Fraction as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
CERT = ROOT / "certificates" / "EXACT_1_5_n7_T5.json"
CHECKERS = [ROOT / "certificates" / "verify_certificate.py", ROOT / "certificates" / "independent_check.py"]

C0 = json.load(open(CERT))
n, T = C0["n"], C0["T"]
N = 2 ** n
bits = [[(x >> i) & 1 for i in range(n)] for x in range(N)]
wt = [sum(b) for b in bits]


def keyS(x, y):
    a, b = sorted((wt[x], wt[y]))
    return f"{a},{b},{sum(p & q for p, q in zip(bits[x], bits[y]))}"


def keyP(x, y):
    tx, ty = (bits[x][0], wt[x] - bits[x][0]), (bits[y][0], wt[y] - bits[y][0])
    u, w = sorted((tx, ty))
    return f"{u[0]},{u[1]}|{w[0]},{w[1]}|{sum(p & q for p, q in zip(bits[x][1:], bits[y][1:]))}"


def swap(x, i):
    return x ^ 1 ^ (1 << i) if (x & 1) != ((x >> i) & 1) else x


def lin_consistent_perturb(C, t, K, d):
    """M1[t][K] += d, compensated in M0[t] and in the next layer (or Gamma0) so all linear
    conditions still hold exactly; only positive semidefiniteness can then fail."""
    C = copy.deepcopy(C)
    D = [[1 if keyP(x, y) == K else 0 for y in range(N)] for x in range(N)]
    S, R = {}, {}
    for x in range(N):
        for y in range(N):
            s = sum(D[swap(x, i)][swap(y, i)] for i in range(n))
            r = sum((1 - 2 * bits[x][i]) * (1 - 2 * bits[y][i]) * D[swap(x, i)][swap(y, i)] for i in range(n))
            k = keyS(x, y)
            if S.setdefault(k, s) != s or R.setdefault(k, r) != r:
                return None
    C["M1"][t][K] = str(F(C["M1"][t].get(K, "0")) + d)
    for k, s in S.items():
        if s:
            C["M0"][t][k] = str(F(C["M0"][t].get(k, "0")) - d * s)
    nxt = C["Gamma0"] if t == T - 1 else C["M0"][t + 1]
    for k in R:
        if R[k] - S[k]:
            nxt[k] = str(F(nxt.get(k, "0")) + d * (R[k] - S[k]))
    return C


def cases():
    yield "original (must PASS)", C0, True
    for acc in ([1, 4], [2, 5], [2, 6], [1], [5], [1, 5, 6], [0, 1, 5]):
        C = copy.deepcopy(C0); C["accept"] = acc
        yield f"wrong function accept={acc}", C, False
    C = copy.deepcopy(C0); C["T"] = T - 1; C["M0"] = C["M0"][:-1]; C["M1"] = C["M1"][:-1]
    yield "one query fewer (drop last layer)", C, False
    C = copy.deepcopy(C0); C["T"] = T - 1; C["M0"] = C["M0"][1:]; C["M1"] = C["M1"][1:]
    yield "one query fewer (drop first layer)", C, False
    C = copy.deepcopy(C0); C["Gamma0"], C["Gamma1"] = C["Gamma1"], C["Gamma0"]
    yield "Gamma0 and Gamma1 swapped", C, False
    for K in ["1,1,0", "1,5,1", "0,1,0"]:
        for d in (F(1, 10 ** 12), F(-1, 10 ** 12)):
            C = copy.deepcopy(C0)
            C["Gamma0"][K] = str(F(C["Gamma0"].get(K, "0")) + d)
            C["Gamma1"][K] = str(F(C["Gamma1"].get(K, "0")) - d)
            yield f"Gamma0<->Gamma1 shift {K} by {d}", C, False
    random.seed(1)
    keys = sorted(C0["M1"][T - 1].keys())
    for t in (T - 1, 2):
        for K in random.sample(keys, 4):
            for d in (F(1, 10 ** 9), F(-1, 10 ** 9)):
                C = lin_consistent_perturb(C0, t, K, d)
                if C is not None:
                    yield f"linear-consistent M1[{t}][{K}] {'+' if d > 0 else '-'}1e-9", C, False


def main():
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for i, (name, C, expect_valid) in enumerate(cases()):
            p = pathlib.Path(tmp) / f"case{i}.json"
            json.dump(C, open(p, "w"))
            verdicts = []
            for chk in CHECKERS:
                r = subprocess.run([sys.executable, str(chk), str(p)], capture_output=True, text=True)
                verdicts.append(r.returncode == 0)
            ok = all(v == expect_valid for v in verdicts)
            bad += not ok
            print(f"{'ok ' if ok else 'BAD'} {name:45s} verifier: {'accept' if verdicts[0] else 'reject'}"
                  f"   independent: {'accept' if verdicts[1] else 'reject'}", flush=True)
    print(f"\n{'ALL CASES BEHAVED AS EXPECTED' if not bad else f'{bad} CASE(S) MISBEHAVED'}")
    return bad == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
