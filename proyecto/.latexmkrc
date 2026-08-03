# Asegura que bibtex/biber encuentren mi_bibliografia.bib y los .bst
# incluso cuando latexmk compila hacia un -output-directory distinto
# (p. ej. build/%DOCFILE%, como hace la extension LaTeX Workshop de VS Code).
ensure_path('BIBINPUTS', '.');
ensure_path('BSTINPUTS', '.');
