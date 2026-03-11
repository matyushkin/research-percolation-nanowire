# Percolation in Nanowire Networks

**Status:** In progress
**Target journal:** npj Computational Materials → Physical Review Letters → Physical Review E
**arXiv:** —
**DOI:** —

## One-line summary

Computational study of percolation and conductivity in random nanowire networks,
focusing on bridge percolation geometry, junction resistance crossover, and hybrid AgNW+QD systems.

## Structure

```
paper/          LaTeX source
  main.tex      Journal-agnostic content
  revtex/       APS wrapper (PRL, PRE, PRB)
  rsc/          RSC wrapper (Nanoscale, Digital Discovery)
  nature/       Nature Portfolio wrapper (npj, Nat. Commun.)
  si.tex        Supplementary Information
  references.bib
notebooks/      Jupyter notebooks (exploration → production)
src/            Python package (reusable code)
data/
  raw/          Original data, never modified
  processed/    Cleaned / computed results
figures/        Final publication figures (PDF + SVG source)
tests/          Unit tests for src/
```

## Reproduce results

```bash
uv sync
uv run jupyter notebook notebooks/
```

## Citation

```bibtex
@article{Matyushkin2026,
  author  = {Matyushkin, Lev B.},
  title   = {...},
  journal = {...},
  year    = {2026},
}
```
