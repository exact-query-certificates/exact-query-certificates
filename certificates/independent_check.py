"""Second, independently written checker (fast; needs `pip install python-flint`).

Usage: python independent_check.py CERT.json

Checks the same Barnum-Saks-Szegedy conditions as verify_certificate.py, but shares no code with
it and decides positive semidefiniteness by a different exact method:

  A real symmetric d x d matrix A is PSD  <=>  the coefficients c_k of det(tI - A) satisfy
  (-1)^(d-k) c_k >= 0 for all k.
  (=>) det(tI - A) = prod (t - lambda_i), so (-1)^(d-k) c_k = e_{d-k}(lambda) >= 0.
  (<=) then (-1)^d det(-sI - A) = sum_k (-1)^(d-k) c_k s^k >= s^d > 0 for s > 0, so there is no
       negative root, and all roots are real because A is symmetric.

The characteristic polynomial is computed exactly over the rationals by FLINT (C library).
Typical time: n = 7 a few seconds, n = 8 under a minute, n = 9 about ten minutes.
"""
import json
import sys
import time

import flint


def parse(s):
    num, _, den = str(s).partition("/")
    return flint.fmpq(int(num), int(den or 1))


def is_psd(rows):
    d = len(rows)
    for i in range(d):
        for j in range(i + 1, d):
            if rows[i][j] != rows[j][i]:
                return False
    c = flint.fmpq_mat(rows).charpoly().coeffs()  # c[0] + c[1] t + ... + c[d] t^d
    c = c + [flint.fmpq(0)] * (d + 1 - len(c))
    return all(c[k] * (-1) ** (d - k) >= 0 for k in range(d + 1))


def check(path):
    C = json.load(open(path))
    n, T, acc = int(C["n"]), int(C["T"]), sorted(int(w) for w in C["accept"])
    if len(C["M0"]) != T or len(C["M1"]) != T or not all(0 <= w <= n for w in acc):
        print("malformed certificate")
        return False
    N = 1 << n
    bit = lambda x, i: (x >> i) & 1  # coordinate i+1 of the input is bit i of the index
    weight = [bin(x).count("1") for x in range(N)]

    def orbit_full(x, y):  # S_n orbit of the pair (x, y)
        lo, hi = min(weight[x], weight[y]), max(weight[x], weight[y])
        return "%d,%d,%d" % (lo, hi, bin(x & y).count("1"))

    def orbit_fix1(x, y):  # orbit under permutations fixing coordinate 1
        p = sorted([(bit(x, 0), weight[x] - bit(x, 0)), (bit(y, 0), weight[y] - bit(y, 0))])
        return "%d,%d|%d,%d|%d" % (p[0][0], p[0][1], p[1][0], p[1][1], bin((x & y) >> 1).count("1"))

    def transpose_coords(x, i):  # exchange coordinates 1 and i+1
        if bit(x, 0) == bit(x, i):
            return x
        return x ^ 1 ^ (1 << i)

    def matrix(table, orbit):
        vals = {k: parse(v) for k, v in table.items()}
        return [[vals.get(orbit(x, y), flint.fmpq(0)) for y in range(N)] for x in range(N)]

    phase = [[1 - 2 * bit(x, i) for i in range(n)] for x in range(N)]
    good = True
    gram = flint.fmpq_mat([[1] * N for _ in range(N)])  # J: Gram matrix of the initial states
    for t in range(T):
        A0 = matrix(C["M0"][t], orbit_full)
        A1 = matrix(C["M1"][t], orbit_fix1)
        parts = [A0] + [[[A1[transpose_coords(x, i)][transpose_coords(y, i)] for y in range(N)]
                         for x in range(N)] for i in range(n)]
        total = flint.fmpq_mat([[sum((P[x][y] for P in parts), flint.fmpq(0)) for y in range(N)] for x in range(N)])
        lin = total == gram
        p0, p1 = is_psd(A0), is_psd(A1)
        print(f"  before query {t + 1}: sum of parts = previous Gram {lin}; PSD {p0} {p1}", flush=True)
        good = good and lin and p0 and p1
        gram = flint.fmpq_mat([[parts[0][x][y] + sum((phase[x][i] * phase[y][i] * parts[i + 1][x][y]
                                                      for i in range(n)), flint.fmpq(0))
                                for y in range(N)] for x in range(N)])
    G0, G1 = matrix(C["Gamma0"], orbit_full), matrix(C["Gamma1"], orbit_full)
    lin = (flint.fmpq_mat(G0) + flint.fmpq_mat(G1)) == gram
    p0, p1 = is_psd(G0), is_psd(G1)
    exact = all((G0 if weight[x] in acc else G1)[x][x] == 0 for x in range(N))
    print(f"  output: Gamma0 + Gamma1 = final Gram {lin}; PSD {p0} {p1}; never wrong {exact}")
    good = good and lin and p0 and p1 and exact
    return good


if __name__ == "__main__":
    t0 = time.time()
    ok = check(sys.argv[1])
    print(("INDEPENDENT CHECK PASSED" if ok else "INDEPENDENT CHECK FAILED") + f"  ({time.time() - t0:.0f} s)")
    sys.exit(0 if ok else 1)
