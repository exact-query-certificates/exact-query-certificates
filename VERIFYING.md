# How to check these results

The claim: for each certificate `EXACT_k_l_nN_TT.json`, there is a zero-error quantum algorithm
computing EXACT<sub>k,l</sub><sup>n</sup> with T queries, and T equals the known lower bound
max{n−k, l} − 1 (Ambainis–Iraids–Nagaj 2017, Theorem 5). So Q<sub>E</sub>(EXACT<sub>k,l</sub><sup>n</sup>) = T.

You should not need to trust anything in this repository except a short checker you can read.

## Option 1: look at GitHub's run (no installation)

Open the **Actions** tab of this repository. On every push, GitHub's servers:

- check the SHA-256 hashes of the certificate files;
- run the **independent checker** on all nine certificates;
- run the **first checker** on the n ≤ 8 certificates (the n = 9 ones take hours with it; trigger the
  workflow manually with "full" ticked to include them);
- run the tamper test: 33 deliberately broken certificates, all of which must be rejected.

## Option 1b: run it in your browser

[Open `verify.ipynb` in Google Colab](https://colab.research.google.com/github/exact-query-certificates/exact-query-certificates/blob/main/verify.ipynb) and choose Runtime → Run all.

## Option 2: run it yourself (one command)

```bash
git clone https://github.com/exact-query-certificates/exact-query-certificates
cd exact-query-certificates
pip install python-flint          # optional, but makes it ~10-100x faster
python verify_all.py --quick      # n = 7, 8: a few minutes
python verify_all.py              # all nine: about half an hour more for n = 9
python tests/test_verifier_rejects.py   # the checkers reject 33 broken certificates
```

Or a single certificate:

```bash
python certificates/independent_check.py certificates/EXACT_1_5_n7_T5.json   # ~1 s
python certificates/verify_certificate.py certificates/EXACT_1_5_n7_T5.json  # ~10 s, standard library only
```

## Option 3: read the checker (about 100 lines)

The certificate gives rational matrices; the checker rebuilds every 2<sup>n</sup>×2<sup>n</sup> matrix and tests the
conditions of Barnum–Saks–Szegedy (2003). In `certificates/verify_certificate.py`:

| condition (BSS) | meaning | code |
|---|---|---|
| file is well formed | n, T, accept set and number of layers are sane | lines 74–76 |
| Σ<sub>i</sub> M<sub>0</sub><sup>(i)</sup> = J | the starting state does not depend on the input | `prev` starts as all-ones (line 115), compared at line 120 for t = 0 |
| Σ<sub>i</sub> M<sub>t</sub><sup>(i)</sup> = Σ<sub>i</sub> E<sub>i</sub> ∘ M<sub>t−1</sub><sup>(i)</sup> | each oracle call followed by a fixed unitary is consistent | line 124 builds Σ E<sub>i</sub>∘M, line 120 compares |
| every M<sub>t</sub><sup>(i)</sup> PSD | these are genuine Gram matrices | line 121 (M<sup>(i)</sup>, i ≥ 2, are row/column permutations of M<sup>(1)</sup>) |
| Γ<sup>0</sup> + Γ<sup>1</sup> = final Gram matrix, both PSD | the final measurement exists | lines 126–128 |
| Γ<sup>1−f(x)</sup>[x,x] = 0 for all x | the probability of a wrong answer is exactly 0 | line 129 |
| T = max{n−k, l} − 1 | matches the published lower bound | line 136 |

`psd()` (line 43) is exact symmetric Gaussian elimination: a negative pivot means not PSD, and a zero
pivot is allowed only if the rest of its row is zero.

`certificates/independent_check.py` checks the same conditions with separately written code and a
different PSD test (signs of the characteristic polynomial, computed exactly by FLINT). The two share
no code.

Why passing these checks proves the claim (Barnum–Saks–Szegedy, the direction needed): write each
PSD matrix as a Gram matrix of vectors. The vectors for "before query t+1" and "after query t" then have
equal inner products for every pair of inputs, so a single unitary, the same for all inputs, maps
one set to the other. The first condition makes the initial state input-independent, and the last makes
the final measurement error-free. This works in the query model of Ambainis–Iraids–Nagaj (phase oracle
with a null query), which is equivalent query-for-query to the standard XOR oracle.

## If you find a problem

Please open an issue. A failing check, a mistake in the reduction from the BSS conditions to the
claim, or prior work that already covers these cases would all be very welcome.
