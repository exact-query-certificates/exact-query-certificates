"""Check every certificate in certificates/ and print a summary.

    pip install python-flint      # optional but strongly recommended (fast exact arithmetic)
    python verify_all.py          # both checkers on n <= 8, independent checker on all (~15-60 min for n = 9)
    python verify_all.py --quick  # n <= 8 only (a few minutes)
    python verify_all.py --full   # also run the slow pure-Python checker on n = 9 (hours)

Two checkers, written separately and sharing no code:
  certificates/verify_certificate.py  pure Python (fractions), PSD by exact symmetric elimination
  certificates/independent_check.py   python-flint, PSD by the signs of the characteristic polynomial
"""
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
CERTS = sorted((ROOT / "certificates").glob("EXACT_*.json"), key=lambda p: (json.load(open(p))["n"], p.name))


def run(script, cert):
    t0 = time.time()
    r = subprocess.run([sys.executable, str(ROOT / "certificates" / script), str(cert)], capture_output=True, text=True)
    return r.returncode == 0, time.time() - t0, (r.stdout + r.stderr).strip().splitlines()[-1:]


def main():
    quick, full = "--quick" in sys.argv, "--full" in sys.argv
    try:
        import flint  # noqa: F401
        have_flint = True
    except ImportError:
        have_flint = False
        print("python-flint not installed: running only the pure-Python checker (slower).\n")
    rows, all_ok = [], True
    for cert in CERTS:
        C = json.load(open(cert))
        n, T, acc = C["n"], C["T"], sorted(C["accept"])
        if quick and n > 8:
            continue
        k, l = acc
        lower = max(n - k, l) - 1
        res = {}
        if have_flint:
            res["independent"] = run("independent_check.py", cert)
        if n <= 8 or full or not have_flint:
            res["verifier"] = run("verify_certificate.py", cert)
        ok = all(v[0] for v in res.values()) and lower == T
        all_ok &= ok
        cells = "  ".join(f"{name}: {'PASS' if v[0] else 'FAIL'} ({v[1]:.0f}s)" for name, v in res.items())
        line = f"n={n} accept={{{k},{l}}}  T={T}  lower bound={lower}  ->  {'Q_E = %d' % T if ok else 'NOT ESTABLISHED'}   [{cells}]"
        print(line, flush=True)
        rows.append(line)
    print("\nALL CERTIFICATES VERIFIED" if all_ok else "\nSOME CERTIFICATE FAILED - see above")
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
