import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import stim
from qkit import sim
from qkit.sim import Circuit

rng = np.random.default_rng(1234)


def rand_unitary(d):
    q, r = np.linalg.qr(rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)))
    return q * (np.diag(r) / abs(np.diag(r)))


def random_circuit(n, depth):
    c = Circuit(n)
    for _ in range(depth):
        k = int(rng.integers(1, min(3, n) + 1))
        qs = list(rng.permutation(n))
        targets, rest = qs[:k], qs[k:]
        nc = int(rng.integers(0, len(rest) + 1)) if rest else 0
        controls = [(int(q), int(rng.integers(0, 2))) for q in rest[:nc]]
        c.gate(rand_unitary(2 ** k), [int(t) for t in targets], controls)
    return c


def test_kernel_vs_bruteforce():
    worst = 0
    for n in range(1, 6):
        for _ in range(20):
            c = random_circuit(n, 12)
            worst = max(worst, np.abs(c.unitary() - sim.unitary_bruteforce(c)).max())
    assert worst < 1e-10, worst
    return worst


def test_clifford_vs_stim():
    names = ["H", "S", "X", "Y", "Z", "CX", "CZ", "SWAP"]
    worst = 1.0
    for trial in range(200):
        n = int(rng.integers(1, 9))
        c, s = Circuit(n), stim.Circuit()
        for _ in range(40):
            g = names[int(rng.integers(0, len(names) if n > 1 else 5))]
            if g in ("CX", "CZ", "SWAP"):
                a, b = (int(v) for v in rng.choice(n, 2, replace=False))
                {"CX": c.cx, "CZ": c.cz, "SWAP": c.swap}[g](a, b)
                s.append(g, [a, b])
            else:
                q = int(rng.integers(0, n))
                getattr(c, g)(q)
                s.append(g, [q])
        ts = stim.TableauSimulator()
        ts.do_circuit(s)
        ref = ts.state_vector(endian="little")
        if len(ref) < 2 ** n:  # stim only tracks qubits it has seen
            ref = np.kron(sim.zero_state(n - int(np.log2(len(ref)))), ref)
        worst = min(worst, sim.fidelity_up_to_phase(c.run(), ref))
    assert worst > 1 - 1e-9, worst
    return worst


def test_density_matrix():
    n = 3
    c = random_circuit(n, 15)
    rho = sim.dm_from_state(sim.zero_state(n))
    for op in c.ops:
        rho = sim.dm_apply_op(rho, n, op)
    err1 = np.abs(rho - sim.dm_from_state(c.run())).max()
    # depolarizing on one qubit of a Bell pair: rho -> (1-p) rho + p (I/2 (x) Tr_0 rho)
    b = Circuit(2).H(0).cx(0, 1)
    r0 = sim.dm_from_state(b.run())
    p = 0.3
    r1 = sim.dm_kraus(r0, 2, sim.depolarize1(p), [0])
    red = np.einsum("aibi->ab", r0.reshape(2, 2, 2, 2))  # trace out qubit 0 (low bit)
    expect = (1 - p) * r0 + p * np.kron(red, np.eye(2) / 2)
    err2 = np.abs(r1 - expect).max()
    assert err1 < 1e-10 and err2 < 1e-10 and abs(np.trace(r1) - 1) < 1e-12
    return err1, err2


def qft(n):
    c = Circuit(n)
    for j in reversed(range(n)):
        c.H(j)
        for k in reversed(range(j)):
            c.gate(sim.phase(np.pi / 2 ** (j - k)), j, [k], f"P(pi/{2 ** (j - k)})")
    for j in range(n // 2):
        c.swap(j, n - 1 - j)
    return c


def test_qft():
    worst = 0
    for n in range(1, 7):
        N = 2 ** n
        dft = np.exp(2j * np.pi * np.outer(range(N), range(N)) / N) / np.sqrt(N)
        worst = max(worst, np.abs(qft(n).unitary() - dft).max())
    assert worst < 1e-10, worst
    return worst


def test_quirk_export():
    c = Circuit(3).H(0).cx(0, 1).gate(sim.ry(0.3), 2, [(0, 0)]).swap(0, 2).T(1)
    url = c.quirk_url()
    assert url.startswith("https://algassert.com/quirk#circuit=")
    return url


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(f"{name}: {fn()}")
