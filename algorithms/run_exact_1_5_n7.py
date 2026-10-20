"""Explicit exact 5-query quantum algorithm for EXACT_{1,5}^7 (f(x)=1 iff |x| in {1,5}),
run gate-by-gate in qkit.sim on all 128 inputs.

Space: query register q in {0..7} (q=0: no query) x workspace w in {0,1,2} x output o in {0,1};
basis index = q*6 + w*2 + o  (48 dims), embedded in 6 qubits (64 dims; unused 16 dims idle).
Algorithm: U_5 O_x U_4 O_x U_3 O_x U_2 O_x U_1 O_x U_0 |0>, then measure o (qubit 0).
O_x multiplies basis states with q=i>=1 by (-1)^{x_i}. Found by numerical search, polished by
Levenberg-Marquardt to machine precision (results/ex15_7_T5_w3.log)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from qkit.sim import Circuit, probs

d = np.load(pathlib.Path(__file__).with_name("EXACT_1_5_n7_T5_compact.npz"))
U, n, w = d["U"], int(d["n"]), int(d["w"])
D = (n + 1) * w * 2
assert U.shape == (6, D, D) and D == 48

def embed(M):
    E = np.eye(64, dtype=complex); E[:D, :D] = M
    return E

def oracle(x):
    q = np.arange(64) // (2 * w)
    ph = np.ones(64)
    for i in range(1, n + 1):
        ph[(q == i) & (np.arange(64) < D)] = (-1) ** ((x >> (i - 1)) & 1)
    return np.diag(ph).astype(complex)

worst = 0.0
for x in range(2 ** n):
    c = Circuit(6)
    c.gate(embed(U[0]), range(6), name="U0")
    for t in range(1, 6):
        c.gate(oracle(x), range(6), name="O_x").gate(embed(U[t]), range(6), name=f"U{t}")
    p1 = probs(c.run(), [0])[1]          # output bit o = qubit 0 (lowest index bit)
    fx = int(bin(x).count("1") in (1, 5))
    worst = max(worst, p1 if fx == 0 else 1 - p1)
print(f"EXACT_{{1,5}}^7 with 5 queries: max error probability over all 128 inputs = {worst:.2e}")
print("unitarity:", max(np.abs(u.conj().T @ u - np.eye(D)).max() for u in U))
