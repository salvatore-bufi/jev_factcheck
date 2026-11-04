#!/bin/sh
# Build the whole documentation as one PDF: pdf/fine_tuning_factcheck_evidence_implementazione.pdf
# Needs pandoc and xelatex. Run from anywhere: sh Documentation/fine_tuning_factcheck_evidence/pdf/build.sh
# Images are kept in the text flow (no floats: -implicit_figures) and limited in height (pdf/header.tex),
# code lines wrap (fvextra): nothing can be pushed off the page.
set -e
cd "$(dirname "$0")/.."
pandoc testa_per_frase/[0-9][0-9]_*.md \
  --resource-path=testa_per_frase \
  --from markdown+tex_math_dollars+pipe_tables-implicit_figures \
  --pdf-engine=xelatex -H pdf/header.tex \
  -V mainfont="DejaVu Serif" -V sansfont="DejaVu Sans" -V monofont="DejaVu Sans Mono" -V monofontoptions="Scale=0.8" \
  -V geometry:margin=2cm -V papersize=a4 -V fontsize=10pt -V colorlinks=true -V lang=it \
  --toc --toc-depth=2 \
  -M title="Fact checking con evidenza per frase — implementazione (KnowLine-4B)" \
  -M date="2026-11-04" \
  -o pdf/fine_tuning_factcheck_evidence_implementazione.pdf
echo "-> pdf/fine_tuning_factcheck_evidence_implementazione.pdf"
