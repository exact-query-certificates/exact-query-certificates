# Exact quantum query complexity of EXACT<sub>k,l</sub><sup>n</sup>: certified cases of the Ambainis–Iraids–Nagaj conjecture

**TL;DR.** For nine Boolean functions whose exact quantum query complexity was open, this repository
contains machine-checkable **exact rational certificates** that zero-error quantum algorithms with
max{n−k, l} − 1 queries exist. With the known lower bound of Ambainis, Iraids and Nagaj
(2017, Thm. 5) this settles their exact quantum query complexity, in each case agreeing with
their Conjecture 1.

[![verify certificates](https://github.com/exact-query-certificates/exact-query-certificates/actions/workflows/verify.yml/badge.svg)](https://github.com/exact-query-certificates/exact-query-certificates/actions/workflows/verify.yml)

**Check it yourself** — no need to trust this repository, only a ~100-line checker
(details and a line-by-line map of what is checked: [VERIFYING.md](VERIFYING.md)):

```bash
git clone https://github.com/exact-query-certificates/exact-query-certificates && cd exact-query-certificates
pip install python-flint          # needed by independent_check.py; verify_certificate.py needs nothing
python certificates/independent_check.py certificates/EXACT_1_5_n7_T5.json  # ~1 s
python verify_all.py --quick                                              # all n <= 8, two independent checkers
python tests/test_verifier_rejects.py                                     # both reject 33 broken certificates
```

Or just open the **Actions** tab: GitHub's servers re-run all of this on every push.

*Status: not peer reviewed. Author: Anonymous. The search, the certificates and
the code were produced with substantial help from an AI assistant (Claude, by Anthropic); every
claim below is backed by a certificate that can be checked independently of that process.*

---

## The problem

EXACT<sub>k,l</sub><sup>n</sup>(x) = 1 iff the Hamming weight |x| ∈ {k, l}, for x ∈ {0,1}<sup>n</sup>.
Q<sub>E</sub>(f) is the minimum number of queries of a quantum algorithm that computes f with
certainty (phase oracle O<sub>x</sub>|i⟩ = (−1)<sup>x<sub>i</sub></sup>|i⟩, O<sub>x</sub>|0⟩ = |0⟩, arbitrary workspace).

Ambainis, Iraids and Nagaj ([arXiv:1608.02374](https://arxiv.org/abs/1608.02374)) proved
max{n−k, l} − 1 ≤ Q<sub>E</sub>(EXACT<sub>k,l</sub><sup>n</sup>) ≤ max{n−k, l} + 1, settled d = l − k ∈ {2, 3}, and
conjectured that Q<sub>E</sub> = max{n−k, l} − 1 whenever d ≥ 2 (Conjecture 1). Yao and Ye
([arXiv:2303.10935](https://arxiv.org/abs/2303.10935)) settled k = 0 and (k, l) = (1, n−1).
For d ≥ 4 all other cases were open as far as we could find (see [Prior work](#prior-work)).

## Results

| n | {k, l} | equivalent to | Q<sub>E</sub> (this work) | classical D | previously |
|---|--------|---------------|-----------------|-------------|------------|
| 7 | {1, 5} | {2, 6} | **5** | 7 | open |
| 8 | {1, 5} | {3, 7} | **6** | 8 | open |
| 8 | {1, 6} | {2, 7} | **6** | 8 | open |
| 8 | {2, 6} | – | **5** | 8 | open; observed numerically (Wu et al. 2022), no proof |
| 9 | {2, 6} | {3, 7} | **6** | 9 | open |
| 9 | {2, 7} | – | **6** | 9 | open |
| 9 | {1, 5} | {4, 8} | **7** | 9 | open |
| 9 | {1, 6} | {3, 8} | **7** | 9 | open |
| 9 | {1, 7} | {2, 8} | **7** | 9 | open |

"Equivalent" uses EXACT<sub>k,l</sub><sup>n</sup>(x) = EXACT<sub>n−l,n−k</sub><sup>n</sup>(x̄).
The upper bound in each row is our certificate; the matching lower bound is AIN Theorem 5.

**Numerical evidence beyond the certificates** ([data/sdp_sweep_d4.log](data/sdp_sweep_d4.log)):
a symmetry-reduced SDP was solved for every instance with d ≥ 4 and 6 ≤ n ≤ 10 (47 instances up to
complement symmetry, 17 of them open). For all 17 open instances the optimal worst-case error is
≥ 2.9·10⁻⁴ with max{n−k, l} − 2 queries and ≤ 1.0·10⁻⁶ (solver accuracy) with max{n−k, l} − 1
queries, consistent with the conjecture. This is numerical, not proof: at n = 10 the solver's
"zero" (~10⁻⁶) is close to the smallest genuinely positive values seen in proven cases
(1.9·10⁻⁶ for n = 10, {0, 10}), so the n = 10 entries in particular are weak evidence.

## What a certificate is, and why it proves the claim

By Barnum, Saks and Szegedy (2003), a T-query algorithm with zero error exists iff there are real
positive semidefinite (PSD) matrices M<sub>t</sub><sup>(i)</sup> (t < T, i = 0..n) and Γ<sup>0</sup>, Γ<sup>1</sup>, indexed by inputs x, y, with

1. Σ<sub>i</sub> M<sub>0</sub><sup>(i)</sup> = J (all-ones);
2. Σ<sub>i</sub> M<sub>t</sub><sup>(i)</sup> = Σ<sub>i</sub> E<sub>i</sub> ∘ M<sub>t−1</sub><sup>(i)</sup> for t ≥ 1, where E<sub>0</sub> = J and E<sub>i</sub>[x,y] = (−1)<sup>x<sub>i</sub>+y<sub>i</sub></sup>;
3. Γ<sup>0</sup> + Γ<sup>1</sup> = Σ<sub>i</sub> E<sub>i</sub> ∘ M<sub>T−1</sub><sup>(i)</sup>;
4. Γ<sup>b</sup>[x,x] = 0 whenever f(x) ≠ b.

(M<sub>t</sub><sup>(i)</sup> is the Gram matrix of the query-register-|i⟩ part of the state before query t+1.
For the direction used here: factor each PSD matrix as a Gram matrix; equal Gram matrices before and
after each oracle call give an isometry, extendable to a unitary, mapping one family of states to
the next; condition 4 makes the final measurement error-free.)

The model is the one of AIN (phase oracle with a "null" query |0⟩); it is equivalent query-for-query
to the standard XOR oracle (map index 0 to |1⟩|+⟩ and index i to |i⟩|−⟩; conversely conjugate the
XOR oracle by H on its target), so the results hold in either model.

A certificate lists these matrices with **exact rational entries**. Because f is symmetric, they
are stored compactly by S<sub>n</sub>-orbit (see the verifier's docstring). The verifier rebuilds every
2<sup>n</sup> × 2<sup>n</sup> matrix and checks 1–4 and PSD-ness in exact arithmetic. It shares no code with
the search. We also convert certificates back into explicit unitaries and run them: worst-case
error 8.9·10⁻¹⁶ (float rounding) on all inputs of the n = 7 case (`qkit/extract.py`).

### Independent review

Before release, an independent reviewer (a separate AI agent instructed to try to break the claim;
not a human referee) re-derived the certificate ⇒ algorithm direction, checked AIN Theorem 5 and the
complement symmetry, fuzzed the exact PSD routine against a characteristic-polynomial test (3000
matrices, 0 mismatches), re-verified the n = 7 certificate with its own independently written checker,
re-ran the verifier on two n = 8 certificates, and confirmed that 31 adversarially tampered
certificates (other functions, fewer queries, sum-preserving perturbations of 10⁻¹² to 10⁻⁹) are
all rejected. Following that review there are now **two checkers that share no code**:
`certificates/verify_certificate.py` (standard library only, PSD by exact elimination) and
`certificates/independent_check.py` (python-flint, PSD by the signs of the exact characteristic
polynomial), and the tamper test ships as `tests/test_verifier_rejects.py`. The first checker also
prints the AIN lower bound and the conclusion Q<sub>E</sub> = T.

Verification time with the independent checker: n = 7 about 1 s, n = 8 about 10 s, n = 9 four to
six minutes each. The first checker is pure Python and much slower for n = 9 (hours). SHA-256 hashes
of the certificates and checkers are in [certificates/SHA256SUMS](certificates/SHA256SUMS).

## How the certificates were found

1. **Search.** Optimise the T+1 unitaries of a query algorithm directly (`qkit/query.py`), then
   polish with Levenberg–Marquardt; and solve the BSS SDP (`qkit/sdp.py`). For n = 7, {1, 5} both
   find error 0 at T = 5 (the explicit 48-dimensional algorithm in `algorithms/` has worst error
   6·10⁻³¹) and the SDP gives optimal error 0.041 at T = 4.
2. **Symmetry reduction** (`qkit/sdp_sym.py`): average over S<sub>n</sub>; invariant matrices block-diagonalise
   via the (n−j, j) isotypic components of the hypercube, so PSD constraints have size at most 2(n−1) (blocks of sizes 8, 6, 4, 2 and 14, 10, 6, 2 for n = 7) instead of 2<sup>n</sup>.
3. **Facial reduction** (`qkit/certify.py`): exact solutions lie on a face. After t queries the
   amplitudes are polynomials of degree ≤ t, and final amplitudes on output b vanish off f<sup>−1</sup>(b)
   (W<sub>b</sub>). Backwards: a fixed unitary preserves the span of amplitude functions and the oracle
   multiplies register i by χ<sub>i</sub>, so register-i components before query t+1 lie in
   χ<sub>i</sub>Z<sub>t+1</sub> ∩ V<sub>t</sub>, with Z<sub>t</sub> = Σ<sub>i</sub>(χ<sub>i</sub>Z<sub>t+1</sub> ∩ V<sub>t</sub>) and Z<sub>T</sub> = W<sub>0</sub> + W<sub>1</sub>.
4. **Rounding.** Solve for a point of maximal margin inside the face, round it to rationals,
   re-solve all linear constraints exactly (RREF over ℚ, python-flint), and check positive
   definiteness on the face exactly (Sylvester's criterion), then the full conditions as above.

## Reproduce

```bash
pip install -r requirements.txt
python tests/test_sim.py        # simulator vs brute force, vs Stim, QFT == DFT
python tests/test_certify.py    # certifies known cases, refuses an impossible one
python experiments/sdp_open.py 7 1,5 4 6                 # optimal error for T = 4, 5, 6
python experiments/certify_export.py 7 1,5 5 out.json    # ~1 min
python certificates/verify_certificate.py out.json
python algorithms/run_exact_1_5_n7.py                    # run the explicit algorithm on all inputs
```

## Prior work

- A. Ambainis, J. Iraids, D. Nagaj, *Exact quantum query complexity of EXACT<sub>k,l</sub><sup>n</sup>*, SOFSEM 2017, [arXiv:1608.02374](https://arxiv.org/abs/1608.02374): conjecture, lower bound, d ∈ {2, 3}.
- P. Yao, Z. Ye (MOD<sub>m</sub><sup>n</sup> and EXACT<sub>k,l</sub><sup>n</sup>), [arXiv:2303.10935](https://arxiv.org/abs/2303.10935): k = 0 and (1, n−1).
- Z. Wu, S.-Y. Hou, C. Zhang, L. Li, B. Zeng, *Variational learning algorithms for quantum query complexity*, [arXiv:2205.07449](https://arxiv.org/abs/2205.07449): numerical values for even n with k, l centred at n/2 (including n = 8, {2, 6}); no proofs.
- A. Montanaro, R. Jozsa, G. Mitchison, *On exact quantum query complexity*, [arXiv:1111.0475](https://arxiv.org/abs/1111.0475): SDP numerics, all symmetric functions on n ≤ 6.
- H. Barnum, M. Saks, M. Szegedy, *Quantum query complexity and semidefinite programming*, CCC 2003.
- G. Xu et al., *Exact quantum query complexity of the five-bit symmetric Boolean function EXACT<sub>3,4</sub><sup>5</sup>*, [arXiv:2610.24518](https://arxiv.org/abs/2610.24518): a rational *dual* certificate (lower bound) with the same kind of tools.

If you know of earlier work covering any of the cases above, please open an issue.

## Layout

`certificates/` certificates + standalone verifier · `algorithms/` explicit n = 7 algorithm ·
`qkit/` simulator, search, SDPs, certification, extraction · `experiments/` scripts ·
`data/` logs of the runs reported here · `tests/` · `paper/` short write-up.

## License

MIT (see LICENSE).
