# Integrated Whole-Brain Modeling — LaTeX Source

This package contains the reproducible LaTeX source for the SuperCognition
Labs technical thesis and development program:

> *Integrated Whole-Brain Modeling Across Modalities, Scales, and Dynamics for
> Individualized Neurotechnology*

The document uses a neutral SuperCognition Labs layout. It is marked “Draft.
Pending Development,” asserts no completed empirical result, and has no DOI.
This source package corresponds to version V6.

## Build

Run from this directory:

```sh
pdflatex -interaction=nonstopmode -halt-on-error sc_wbd_frontiers.tex
bibtex sc_wbd_frontiers
pdflatex -interaction=nonstopmode -halt-on-error sc_wbd_frontiers.tex
pdflatex -interaction=nonstopmode -halt-on-error sc_wbd_frontiers.tex
```

Build the comprehensive implementation supplement separately:

```sh
pdflatex -interaction=nonstopmode -halt-on-error sc_wbd_supplement.tex
bibtex sc_wbd_supplement
pdflatex -interaction=nonstopmode -halt-on-error sc_wbd_supplement.tex
pdflatex -interaction=nonstopmode -halt-on-error sc_wbd_supplement.tex
```

The six diagrams are native vector TikZ figures defined in figures.tex; all
scientific block diagrams use square corners and mitered connectors. The main
thesis, compiler refusals, identifiability benchmark, evidence contract,
worked TMS case, and agent-facing build order are in thesis_contract.tex. The
scientific program is in body.tex. supplementary_summary.tex keeps a compact
source-role appendix in the main PDF. appendix.tex is compiled into the
separate implementation supplement and contains the full acquisition and
calibration register, source-card and gradient contract,
coordinate/clock/calibration manifest, and heterogeneous-mixture leakage and
evaluation protocol. Citations are in references.bib.

TRIBE v2 is quarantined as an optional, off-by-default interface distillation
experiment, not participant data, a subject likelihood, or an empirical
observation. Dataset names in the
appendix are development candidates whose use remains conditional on version,
license, consent, governance, preprocessing, and validation checks.
