#The Polyamine CIRBP axis and Inflammaging
Master Thesis: A Reproducible Whole-Blood Transcriptomic Analysis of a Curated Gene Panel 
# The Polyamine–CIRBP Axis and Inflammaging

**Master's Thesis (TFM) — Bioinformatics MSc, UNIR (with CNIO), 2025–2026**
*A Reproducible Whole-Blood Transcriptomic Analysis of a Curated Gene Panel*

Author: Alejandra Martín Sevilla · ORCID [0009-0001-0186-7680](https://orcid.org/0009-0001-0186-7680)
Director: Leonardo Garma

---

## Overview

`tfm_pipeline.py` implements Methods 3.2–3.9 of the thesis and regenerates the tables and
figures of Results 4.1–4.13 (plus Annex F) from public data.

It tests whether a curated **106-gene panel** tracks chronological age and inflammaging in
human blood. The panel covers polyamine metabolism, eIF5A hypusination, the CIRBP axis,
SASP, NF-κB targets and senescence markers.

| Cohort | Tissue / assay | Age | Panel |
|---|---|---|---|
| **GTEx v11** | Whole blood, bulk RNA-seq (TPM) | Decade bins | 106 genes |
| **GSE233321** | Blood, scRNA-seq → donor pseudobulk (≥18 y) | Exact years | 106 genes |
| **OneK1K** | PBMC, scRNA-seq → donor pseudobulk | Exact years | 87 genes |

### Analyses

1. Per-gene age association (OLS, sex-adjusted, BH-FDR 0.05)
2. Axis sub-scores (polyamine, hypusination, CIRBP module, SASP, PCAS)
3. Age clocks — Elastic Net and Random Forest, 10-fold CV
4. Random Forest hyperparameter sweep and grid search
5. Feature importances, machinery–inflammation coupling, bootstrap CIs
6. Cell-type deconvolution (NNLS, LM22) and immune composition vs age
7. Cross-cohort concordance, ROC/PR age-group classification, stepwise models
8. Dataset-bias diagnostics (Annex F)

---

## Repository structure

```
.
├── README.md
├── LICENSE              # MIT
├── tfm_pipeline.py      # full pipeline
├── data/                # raw inputs (not tracked; download from sources below)
└── results/             # generated output
    ├── tables/
    ├── figures/
    └── run_summary.json
```

## Input data

| File | Source |
|---|---|
| GTEx v11 whole-blood gene TPM (`.gct.gz`) + `SubjectPhenotypesDS.txt` | [GTEx Portal](https://gtexportal.org/home/downloads) |
| GSE233321 single-cell object (`.h5ad`) | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE233321) |
| OneK1K single-cell object (`.h5ad`) | [CELLxGENE](https://cellxgene.cziscience.com/) |
| LM22 signature matrix (optional) | [CIBERSORTx](https://cibersortx.stanford.edu/) |

Cohorts missing from the config are skipped, so GTEx can run on its own.

## Usage

```bash
pip install numpy pandas scipy h5py statsmodels scikit-learn matplotlib xgboost
python tfm_pipeline.py --config config.json --out results
```

`config.json` holds the paths to the input files:

```json
{
  "gtex_gct": "data/gtex/whole_blood_tpm.gct.gz",
  "gtex_phenotypes": "data/gtex/SubjectPhenotypesDS.txt",
  "lm22": "data/reference/LM22.txt",
  "gse233321": { "h5ad": "data/gse233321/GSE233321.h5ad" },
  "onek1k":    { "h5ad": "data/onek1k/onek1k.h5ad" }
}
```

Optional flags: `--skip-rf-sweep`, `--skip-stepwise`.

## Reproducibility

Seed 42 · 10-fold CV · 2,000 bootstrap resamples · FDR α = 0.05.
Table and figure numbers in `results/` match the manuscript.

## Citation

> Martín Sevilla, A. (2026). *The Polyamine–CIRBP Axis and Inflammaging: A Reproducible
> Whole-Blood Transcriptomic Analysis of a Curated Gene Panel*. Master's Thesis, UNIR.

## License

Code under the MIT License. Datasets remain under their original terms of use.
