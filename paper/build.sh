#!/usr/bin/env bash
# Build the manuscript and fail on anything a reviewer would catch:
# undefined references, missing citations, or an unresolved number macro.
# Only the final pass is inspected; earlier passes legitimately warn.
set -uo pipefail
cd "$(dirname "$0")"
run() { pdflatex -interaction=nonstopmode "$@" main.tex >"$1.log" 2>&1; }

pdflatex -interaction=nonstopmode -halt-on-error main.tex >pass1.log 2>&1 \
  || { echo "FAIL: pdflatex errored on the first pass"; tail -30 pass1.log; exit 1; }
bibtex main >bibtex.log 2>&1 || { echo "FAIL: bibtex"; tail -20 bibtex.log; exit 1; }
pdflatex -interaction=nonstopmode main.tex >pass2.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error main.tex >final.log 2>&1 \
  || { echo "FAIL: pdflatex errored on the final pass"; tail -30 final.log; exit 1; }

LOG=final.log
status=0
check() { # pattern, message
  if grep -q "$1" "$LOG"; then
    echo "FAIL: $2"; grep -n "$1" "$LOG" | sed 's/^/    /' | sort -u | head -15; status=1
  fi
}
check "LaTeX Warning: There were undefined references" "undefined references"
check "Citation .* undefined" "undefined citations"
check "Undefined control sequence" "unresolved macro (a number not emitted by make_numbers.py)"
check "There were multiply-defined labels" "duplicate labels"
[ -f main.pdf ] || { echo "FAIL: no PDF produced"; status=1; }

if grep -q "Warning--I didn't find a database entry" bibtex.log; then
  echo "FAIL: bibtex could not resolve:"; grep "didn't find" bibtex.log | sed 's/^/    /'; status=1
fi

if [ "$status" -eq 0 ]; then
  echo "OK: $(pdfinfo main.pdf | awk '/^Pages/{print $2}') pages, $(grep -c newcommand generated_numbers.tex) number macros available"
  echo "PENDING markers for the author: $(cat sections/*.tex main.tex | grep -c 'pending{')"
  uncited=$(grep -o '^@[a-z]*{[^,]*' references.bib | sed 's/.*{//' | while read -r k; do
    grep -q "\\\\cite[a-z]*{[^}]*$k" sections/*.tex || echo "    $k"; done)
  [ -n "$uncited" ] && { echo "NOTE: bibliography entries never cited:"; echo "$uncited"; }
fi
exit "$status"
