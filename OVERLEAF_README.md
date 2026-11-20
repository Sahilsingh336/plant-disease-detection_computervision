# IEEE LaTeX paper

`IEEE_Reasearch_paper.tex` is the manuscript entry point. It uses the IEEEtran conference class, BibTeX with `references.bib`, TikZ vector diagrams from `figures/diagram_sources.tex`, and the result-table source in `tables/experimental_tables.tex`.

## Compile locally

From this directory, run:

```sh
latexmk -pdf IEEE_Reasearch_paper.tex
```

The project uses relative paths and can be uploaded to Overleaf as a folder/ZIP. Select `IEEE_Reasearch_paper.tex` as the main document and allow the standard LaTeX/BibTeX build sequence.

## Before submission

The manuscript explicitly flags unavailable dataset-mirror metadata and missing run-level logs behind two report-level aggregates. Resolve the author-verification notes in the manuscript and table footnotes before treating those entries as independently audited results.