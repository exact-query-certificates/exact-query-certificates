"""Small statevector / density-matrix simulator with Quirk export.

Conventions (match Quirk): qubit 0 is the top wire and the LEAST significant bit
of the basis index. A k-qubit gate on targets (t0, t1, ...) uses t0 as the low
bit of its local index.
"""
import json
import urllib.parse
import numpy as np

SQ2 = 1 / np.sqrt(2)
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) * SQ2
S = np.diag([1, 1j])
T = np.diag([1, np.exp(1j * np.pi / 4)])
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)


def rx(a): return np.array([[np.cos(a / 2), -1j * np.sin(a / 2)], [-1j * np.sin(a / 2), np.cos(a / 2)]])
def ry(a): return np.array([[np.cos(a / 2), -np.sin(a / 2)], [np.sin(a / 2), np.cos(a / 2)]], dtype=complex)
def rz(a): return np.diag([np.exp(-0.5j * a), np.exp(0.5j * a)])
def phase(a): return np.diag([1, np.exp(1j * a)])


# Quirk ids for named single-qubit gates (exported without a custom matrix).
QUIRK_IDS = {"H": "H", "X": "X", "Y": "Y", "Z": "Z", "S": "Z^½", "Sdg": "Z^-½",
             "T": "Z^¼", "Tdg": "Z^-¼", "SX": "X^½"}
NAMED = {"H": H, "X": X, "Y": Y, "Z": Z, "S": S, "Sdg": S.conj().T, "T": T,
         "Tdg": T.conj().T, "SX": np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2}


class Op:
    __slots__ = ("name", "targets", "controls", "matrix")

    def __init__(self, name, targets, controls, matrix):
        self.name, self.targets, self.controls = name, tuple(targets), tuple(controls)
        self.matrix = np.asarray(matrix, dtype=complex)

    def __repr__(self):
        c = "".join(f"{'+' if v else '-'}{q}" for q, v in self.controls)
        return f"{self.name}{list(self.targets)}{'c' + c if c else ''}"


class Circuit:
    def __init__(self, n):
        self.n, self.ops = n, []

    # --- building ---------------------------------------------------------
    def gate(self, matrix, targets, controls=(), name=None):
        """controls: iterable of qubit (control on 1) or (qubit, value)."""
        targets = (targets,) if np.isscalar(targets) else tuple(targets)
        cs = tuple((c, 1) if np.isscalar(c) else (c[0], c[1]) for c in controls)
        m = np.asarray(matrix, dtype=complex)
        assert m.shape == (2 ** len(targets),) * 2, (m.shape, targets)
        used = list(targets) + [q for q, _ in cs]
        assert len(set(used)) == len(used) and all(0 <= q < self.n for q in used), used
        self.ops.append(Op(name or "U", targets, cs, m))
        return self

    def __getattr__(self, nm):
        if nm in NAMED:
            return lambda q, controls=(): self.gate(NAMED[nm], q, controls, nm)
        raise AttributeError(nm)

    def cx(self, c, t): return self.gate(X, t, [c], "X")
    def cz(self, c, t): return self.gate(Z, t, [c], "Z")
    def swap(self, a, b, controls=()): return self.gate(SWAP, (a, b), controls, "SWAP")

    def extend(self, other, offset=0):
        for op in other.ops:
            self.ops.append(Op(op.name, [q + offset for q in op.targets],
                               [(q + offset, v) for q, v in op.controls], op.matrix))
        return self

    def inverse(self):
        c = Circuit(self.n)
        inv = {"S": "Sdg", "Sdg": "S", "T": "Tdg", "Tdg": "T"}
        for op in reversed(self.ops):
            c.ops.append(Op(inv.get(op.name, op.name if op.name in ("H", "X", "Y", "Z", "SWAP") else "U"),
                            op.targets, op.controls, op.matrix.conj().T))
        return c

    # --- simulation -------------------------------------------------------
    def run(self, state=None):
        psi = zero_state(self.n) if state is None else np.array(state, dtype=complex)
        for op in self.ops:
            apply_op(psi, self.n, op)
        return psi

    def unitary(self):
        """Full unitary via the tensordot kernel (column by column)."""
        d = 2 ** self.n
        return np.stack([self.run(np.eye(d, dtype=complex)[k]) for k in range(d)], axis=1)

    # --- export -----------------------------------------------------------
    def quirk_url(self, init=None):
        return quirk_url(self, init)


def zero_state(n):
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1
    return psi


def apply_op(psi, n, op):
    """In place. psi is a flat vector of length 2^n (or 2^n x m batch: flat first axis)."""
    t = psi.reshape((2,) * n + psi.shape[1:])
    ax = lambda q: n - 1 - q
    idx = [slice(None)] * t.ndim
    for q, v in op.controls:
        idx[ax(q)] = v
    sub = t[tuple(idx)]  # basic indexing -> view
    # axes of sub after integer-indexing removes control axes
    removed = sorted(ax(q) for q, _ in op.controls)
    def sub_ax(a): return a - sum(1 for r in removed if r < a)
    k = len(op.targets)
    sub_axes = [sub_ax(ax(q)) for q in reversed(op.targets)]  # high local bit first
    m = op.matrix.reshape((2,) * (2 * k))
    res = np.tensordot(m, sub, axes=(list(range(k, 2 * k)), sub_axes))
    sub[...] = np.moveaxis(res, list(range(k)), sub_axes)


def unitary_bruteforce(circ):
    """Independent reference: build each op as a full 2^n matrix by explicit bit
    manipulation and multiply. O(4^n) per op; only for small n."""
    n, d = circ.n, 2 ** circ.n
    U = np.eye(d, dtype=complex)
    for op in circ.ops:
        F = np.zeros((d, d), dtype=complex)
        k = len(op.targets)
        for col in range(d):
            if any(((col >> q) & 1) != v for q, v in op.controls):
                F[col, col] = 1
                continue
            loc_in = sum(((col >> q) & 1) << j for j, q in enumerate(op.targets))
            base = col
            for q in op.targets:
                base &= ~(1 << q)
            for loc_out in range(2 ** k):
                row = base | sum(((loc_out >> j) & 1) << q for j, q in enumerate(op.targets))
                F[row, col] += op.matrix[loc_out, loc_in]
        U = F @ U
    return U


# --- density matrices (QuEST trick: rho as 2n-qubit vector, ket + conj(bra)) ---
def dm_from_state(psi):
    return np.outer(psi, psi.conj())


def dm_apply_op(rho, n, op):
    """rho flattened row-major: index = row*2^n + col -> row bits are HIGH qubits n..2n-1,
    col bits are LOW qubits 0..n-1. U rho U^dag = U on row qubits, conj(U) on col qubits."""
    v = rho.reshape(-1)
    shift = lambda o, off, conj: Op(o.name, [q + off for q in o.targets],
                                    [(q + off, b) for q, b in o.controls],
                                    o.matrix.conj() if conj else o.matrix)
    apply_op(v, 2 * n, shift(op, n, False))
    apply_op(v, 2 * n, shift(op, 0, True))
    return v.reshape(rho.shape)


def dm_kraus(rho, n, kraus, targets):
    out = np.zeros_like(rho)
    for K in kraus:
        r = rho.copy()
        out += dm_apply_op(r, n, Op("K", targets, (), K))
    return out


def depolarize1(p):
    return [np.sqrt(1 - 3 * p / 4) * I2, np.sqrt(p / 4) * X, np.sqrt(p / 4) * Y, np.sqrt(p / 4) * Z]


# --- utilities ----------------------------------------------------------------
def probs(psi, qubits=None):
    p = np.abs(psi) ** 2
    if qubits is None:
        return p
    n = int(np.log2(len(psi)))
    out = np.zeros(2 ** len(qubits))
    ks = np.arange(len(psi))
    loc = sum(((ks >> q) & 1) << j for j, q in enumerate(qubits))
    np.add.at(out, loc, p)
    return out


def fidelity_up_to_phase(a, b):
    return abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))


def _fmt(z, digits=6):
    r, i = round(z.real, digits), round(z.imag, digits)
    r = 0.0 if r == 0 else r
    i = 0.0 if i == 0 else i
    if i == 0:
        return f"{r:g}"
    if r == 0:
        return f"{i:g}i"
    return f"{r:g}{'+' if i > 0 else '-'}{abs(i):g}i"


def quirk_matrix(m):
    return "{" + ",".join("{" + ",".join(_fmt(z) for z in row) + "}" for row in m) + "}"


def quirk_url(circ, init=None):
    """Quirk JSON: each column is a list indexed by wire. Controls apply to the whole
    column, so controlled ops get their own column; uncontrolled ops are packed."""
    cols, custom, cache = [], [], {}

    def gid(op):
        if op.name in QUIRK_IDS and op.matrix.shape == (2, 2) and np.allclose(op.matrix, NAMED[op.name]):
            return [QUIRK_IDS[op.name]]
        if op.name == "SWAP" and np.allclose(op.matrix, SWAP):
            return ["Swap", "Swap"]
        key = op.matrix.tobytes()
        if key not in cache:
            cache[key] = f"~{len(cache) + 1:x}"
            custom.append({"id": cache[key], "name": op.name, "matrix": quirk_matrix(op.matrix)})
        return [cache[key]]

    def blank():
        return [1] * circ.n

    cur, used = None, set()
    for op in circ.ops:
        ids = gid(op)
        span = sorted(op.targets)
        contiguous = span == list(range(span[0], span[0] + len(span))) and list(op.targets) == span
        if ids[0].startswith("~") and len(op.targets) > 1 and not contiguous:
            raise ValueError(f"custom multi-qubit gate needs contiguous ascending targets: {op}")
        wires = set(op.targets) | {q for q, _ in op.controls}
        if op.controls or cur is None or wires & used:
            cur, used = blank(), set()
            cols.append(cur)
        if ids[0] in ("Swap",):
            for q in op.targets:
                cur[q] = "Swap"
        elif len(op.targets) > 1:
            cur[span[0]] = ids[0]
        else:
            cur[op.targets[0]] = ids[0]
        for q, v in op.controls:
            cur[q] = "•" if v else "◦"
        used |= wires | set(range(span[0], span[-1] + 1))
        if op.controls:
            cur, used = None, set()
    for c in cols:
        while c and c[-1] == 1:
            c.pop()
    doc = {"cols": cols}
    if custom:
        doc["gates"] = custom
    if init:
        doc["init"] = init
    return "https://algassert.com/quirk#circuit=" + urllib.parse.quote(
        json.dumps(doc, ensure_ascii=False, separators=(",", ":")), safe="")
