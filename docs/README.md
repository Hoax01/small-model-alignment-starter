# Assignment Manual

Main source:

```text
docs/assignment_manual.tex
```

Recommended build commands:

```bash
latexmk -pdf docs/assignment_manual.tex
```

or, if `latexmk` is unavailable:

```bash
pdflatex -interaction=nonstopmode docs/assignment_manual.tex
pdflatex -interaction=nonstopmode docs/assignment_manual.tex
```

In VS Code with LaTeX Workshop, open `docs/assignment_manual.tex` and use **Build LaTeX project**.
