#!/bin/bash
# usage: experiments/certify_batch.sh "n:k,l:T n:k,l:T ..."   (run from the repo root)
# env CERT_DENOM_BITS (default 30) sets the rounding denominator 2^bits.
Q="$(cd "$(dirname "$0")/.." && pwd)"
for spec in $1; do
  IFS=: read n acc T <<< "$spec"; tag="EXACT_${acc/,/_}_n${n}_T${T}"
  echo "=== $tag  $(date +%T)"
  python -u "$Q/experiments/certify_export.py" $n $acc $T "$Q/certificates/$tag.json" 2>&1 | grep -v Warn
  [ -f "$Q/certificates/$tag.json" ] && python "$Q/certificates/verify_certificate.py" "$Q/certificates/$tag.json" 2>&1 | tail -1
done
echo "=== batch done $(date +%T)"
