#!/bin/sh
# Compile every figure: src/figXX_*.tex -> ../figXX_*.pdf and ../figXX_*.png (200 dpi).
# Needs xelatex (TikZ, pgfplots) and pdftoppm (poppler). Run from this folder: sh build.sh
set -e
cd "$(dirname "$0")"
for f in fig*.tex; do
  n="${f%.tex}"
  xelatex -interaction=nonstopmode -halt-on-error "$f" > "/tmp/$n.log" 2>&1 || { tail -30 "/tmp/$n.log"; exit 1; }
  mv "$n.pdf" "../$n.pdf"
  pdftoppm -png -r 200 -singlefile "../$n.pdf" "../$n"
  rm -f "$n.aux" "$n.log"
  echo "ok $n"
done
