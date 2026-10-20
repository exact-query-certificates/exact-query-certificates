"""Standalone verifier for exact-quantum-query certificates (pure Python, fractions only).

Usage: python verify_certificate.py CERT.json

Claim checked: there is an exact T-query quantum algorithm (phase oracle O_x|i> = (-1)^{x_i}|i>,
O_x|0> = |0>, arbitrary workspace) computing f(x) = [ |x| in ACCEPT ] on n bits.

By Barnum-Saks-Szegedy (2003) this holds iff there are real PSD matrices indexed by x,y in {0,1}^n:
  M_t^(i), t = 0..T-1, i = 0..n;  Gamma^0, Gamma^1   with
  (1) sum_i M_0^(i) = J (all ones)
  (2) sum_i M_t^(i) = sum_i E_i o M_{t-1}^(i)   (t >= 1), E_0 = J, E_i[x,y] = (-1)^(x_i + y_i)
  (3) Gamma^0 + Gamma^1 = sum_i E_i o M_{T-1}^(i)
  (4) Gamma^b[x,x] = 0 whenever f(x) != b.
(Sketch of the direction used: write each PSD matrix as a Gram matrix of vectors; equal Gram
matrices before/after each query give an isometry, extendable to a unitary, mapping one set
of states to the next; (4) makes the final measurement error-free.)

The certificate stores S_n-symmetric matrices compactly:
  M_t^(0), Gamma^b : value depends on key "a,b,c" = min(|x|,|y|), max(|x|,|y|), |x AND y|
  M_t^(1)          : invariant under permutations fixing coordinate 1 (x_1 = lowest bit);
                     key "p0,w0|p1,w1|c" with (p,w) = (x_1, weight of x_2..x_n), the two
                     pairs sorted, c = |x' AND y'| on coordinates 2..n
  M_t^(i)[x,y] = M_t^(1)[s_i x, s_i y], s_i swapping coordinates 1 and i.
Every matrix entry is rebuilt and all conditions are checked in exact rational arithmetic.
"""
import json
import sys
from fractions import Fraction

try:  # optional fast exact rationals (same semantics); falls back to fractions.Fraction
    import flint

    def Fraction(s=0):  # noqa: F811
        if isinstance(s, str):
            p, _, q = s.partition("/")
            return flint.fmpq(int(p), int(q or 1))
        return flint.fmpq(s)
    BACKEND = "python-flint fmpq"
except ImportError:
    BACKEND = "fractions.Fraction"


def psd(A):
    """Exact PSD test by symmetric Gaussian elimination: a zero pivot must have a zero row."""
    A = [row[:] for row in A]
    d = len(A)
    for i in range(d):
        for j in range(i + 1, d):
            if A[i][j] != A[j][i]:
                return False
    for k in range(d):
        p = A[k][k]
        if p < 0:
            return False
        rk = A[k]
        if p == 0:
            if any(rk[j] != 0 for j in range(k + 1, d)):
                return False
            continue
        nz = [j for j in range(k + 1, d) if rk[j] != 0]
        for i in nz:
            fct = rk[i] / p
            Ai = A[i]
            for j in nz:
                Ai[j] -= fct * rk[j]
    return True


def main(path):
    C = json.load(open(path))
    print(f"exact rational backend: {BACKEND}")
    n, T, accept = C["n"], C["T"], set(C["accept"])
    # structural sanity: malformed files must fail loudly, never pass
    assert isinstance(n, int) and n >= 1 and isinstance(T, int) and T >= 1, "bad n or T"
    assert all(isinstance(w, int) and 0 <= w <= n for w in accept), "accept must be weights in 0..n"
    assert len(C["M0"]) == T and len(C["M1"]) == T, "need exactly T layers of M0 and M1"
    N = 2 ** n
    bits = [[(x >> i) & 1 for i in range(n)] for x in range(N)]
    wt = [sum(b) for b in bits]
    f = [1 if wt[x] in accept else 0 for x in range(N)]
    fr = lambda s: Fraction(s)

    def keyS(x, y):
        a, b = sorted((wt[x], wt[y]))
        return f"{a},{b},{sum(p & q for p, q in zip(bits[x], bits[y]))}"

    def keyP(x, y):
        tx, ty = (bits[x][0], wt[x] - bits[x][0]), (bits[y][0], wt[y] - bits[y][0])
        u, v = sorted((tx, ty))
        c = sum(p & q for p, q in zip(bits[x][1:], bits[y][1:]))
        return f"{u[0]},{u[1]}|{v[0]},{v[1]}|{c}"

    def swap(x, i):
        b0, bi = (x >> 0) & 1, (x >> i) & 1
        return x ^ ((b0 ^ bi) << 0) ^ ((b0 ^ bi) << i) if b0 != bi else x

    missing, unused = [0], [0]

    def build(table, keyf):
        tab = {k: fr(v) for k, v in table.items()}
        seen = set()
        out = []
        for x in range(N):
            row = []
            for y in range(N):
                k = keyf(x, y)
                seen.add(k)
                row.append(tab.get(k, Fraction(0)))  # absent orbit = value 0 (reported below)
            out.append(row)
        missing[0] += len(seen - set(tab))
        unused[0] += len(set(tab) - seen)
        return out

    ok = True
    prev = [[Fraction(1)] * N for _ in range(N)]
    for t in range(T):
        M0 = build(C["M0"][t], keyS)
        M1 = build(C["M1"][t], keyP)
        Ms = [M0] + [[[M1[swap(x, i)][swap(y, i)] for y in range(N)] for x in range(N)] for i in range(n)]
        eq = all(sum(Ms[i][x][y] for i in range(n + 1)) == prev[x][y] for x in range(N) for y in range(N))
        p0, p1 = psd(M0), psd(M1)
        print(f"t={t}: sum_i M_t^(i) equals previous Gram matrix: {eq};  M^(0) PSD: {p0};  M^(1) PSD: {p1}")
        ok &= eq and p0 and p1
        prev = [[Ms[0][x][y] + sum((1 - 2 * bits[x][i - 1]) * (1 - 2 * bits[y][i - 1]) * Ms[i][x][y]
                                   for i in range(1, n + 1)) for y in range(N)] for x in range(N)]
    G0, G1 = build(C["Gamma0"], keyS), build(C["Gamma1"], keyS)
    eq = all(G0[x][y] + G1[x][y] == prev[x][y] for x in range(N) for y in range(N))
    p0, p1 = psd(G0), psd(G1)
    ex = all((G1[x][x] == 0) if f[x] == 0 else (G0[x][x] == 0) for x in range(N))
    print(f"final: Gamma0 + Gamma1 equals final Gram matrix: {eq};  Gamma0 PSD: {p0};  Gamma1 PSD: {p1};  "
          f"zero error on all {N} inputs: {ex}")
    ok &= eq and p0 and p1 and ex
    print(f"orbit keys absent from the file (treated as 0): {missing[0]};  keys in the file never used: {unused[0]}")
    if len(accept) == 2:
        k, l = sorted(accept)
        lb = max(n - k, l) - 1
        print(f"Ambainis-Iraids-Nagaj (2017) Thm 5 lower bound max(n-k, l) - 1 = {lb}; certificate T = {T}"
              + (f"  => Q_E = {T}" if ok and lb == T else ""))
    print(f"\nCERTIFICATE {'VALID' if ok else 'INVALID'}: exact {T}-query algorithm for "
          f"f(x)=[|x| in {sorted(accept)}], n={n}" + ("" if ok else " NOT established"))
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1]) else 1)
