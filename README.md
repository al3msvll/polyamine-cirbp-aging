# The Polyamine–CIRBP Axis and Inflammaging — reproducibility package

Code for the TFM *The Polyamine–CIRBP Axis and Inflammaging: A Reproducible
Whole-Blood Transcriptomic Analysis of a Curated Gene Panel* (UNIR / CNIO,
2025–2026). One script regenerates every table and figure of the Results
chapter from the public inputs below.

## Files

| File | Purpose |
|---|---|
| `tfm_pipeline.py` | The analysis. Sections are numbered after Methods 3.2–3.9. |
| `config.example.json` | Template with the input paths; copy to `config.json` and edit. |
| `smoke_test.py` | Runs the whole pipeline on synthetic inputs in the real file formats. |
| `requirements.txt` | Python dependencies. |
| `TEST_RUN.md` | Output listing of the last full smoke test. |
| `VERIFICATION.md` | Side-by-side comparison of the pipeline's output on the real data with the values in the manuscript. |

## Inputs

| Config key | Source | Notes |
|---|---|---|
| `gtex_gct` | `storage.googleapis.com/adult-gtex/bulk-gex/v11/rna-seq/tpms-by-tissue/gene_tpm_v11_whole_blood.gct.gz` | 74,628 genes × 803 samples |
| `gtex_phenotypes` | `storage.googleapis.com/adult-gtex/annotations/v11/metadata-files/GTEx_Analysis_v11_Annotations_SubjectPhenotypesDS.txt` | decade midpoints are used as the age label |
| `lm22` | CIBERSORT LM22 signature matrix (tab-separated, 547 genes × 22 types) | NNLS deconvolution of GTEx |
| `gse233321.h5ad` | `ftp.ncbi.nlm.nih.gov/geo/series/GSE233nnn/GSE233321/suppl/GSE233321_all_PBMCs.h5ad` (12.5 GB) | obs columns `sample_id`, `Age_in_yrs`, `LS_L2`; raw counts in `raw/X` |
| `onek1k.h5ad` | `datasets.cellxgene.cziscience.com/1e44db10-b572-46cc-adae-dcc7acd44ca6.h5ad` (4.4 GB) | obs columns `donor_id`, `age`, `sex`, `cell_type`; raw counts in `X` |

`obs` entries name the donor, age, sex and cell-type columns of an h5ad
(`config.example.json` holds the right ones for both files); leave them
`null` to let the script detect common names. Raw counts are read from
`raw/X` when present, otherwise from `X`. Large files are read in chunks of
`chunk_cells` cells, so the 12 GB GSE233321 file does not need to fit in memory.
Any cohort whose path is missing from the config is skipped.

## Run

```
pip install -r requirements.txt
python tfm_pipeline.py --config config.json --out results
```

Written for the versions stated in the manuscript (Python 3.10, scikit-learn 1.3)
and tested on Python 3.13 with scikit-learn 1.9, pandas 3.0, numpy 2.5,
xgboost 3.4. `TEST_RUN.md` lists the outputs of the last full test.

`--skip-rf-sweep` skips the Random Forest hyper-parameter exploration (3.7a);
`--skip-stepwise` skips the XGBoost / stacked-ensemble stage (4.13). Both are
the slow parts.

Test the installation without the data:

```
pip install anndata
python smoke_test.py          # about one minute
python smoke_test.py --full   # includes the RF sweep and XGBoost stage
```

## Outputs

`results/tables/` and `results/figures/` are named after the manuscript:
`table01_cohorts.csv` … `table20_GTEx_importances_all.csv`,
`fig04_volcano_GTEx.png` … `fig19_OneK1K_stepwise.png`, `fig21_dataset_bias.png`.
`run_summary.json` holds the scalar results quoted in the text (significant
genes per cohort, PC1 loadings, best RF parameters, concordance statistics).
Out-of-fold predictions for every clock are in `oof_*.csv`.

## Methods as implemented

- Panel: Annex A (106 genes; genes listed in two groups take the last group,
  which reproduces the labels of Annex C). OneK1K uses the panel without the
  polyproline-target group (87 genes).
- Normalisation: log1p-TPM (GTEx), log2-CPM of summed UMI counts (pseudobulk).
  Duplicate symbols are collapsed to the row with the highest mean.
- Age association: OLS `expression ~ age + sex` on z-scored variables,
  Benjamini–Hochberg FDR across the panel; sex is dropped when absent.
- PCAS: PC1 of the four standardised sub-scores, sign fixed once so that it
  correlates positively with PPAS.
- Clocks: ElasticNetCV (l1 ratio grid 0.1–0.9, 40 alphas, inner 5-fold) and
  Random Forest (300 trees, `sqrt`), 10-fold KFold with seed 42, scaler fitted
  on the training folds only, metrics per fold and pooled out-of-fold.
- RF exploration: one-at-a-time sweep over six hyper-parameters, then a grid
  over the influential ones, all scored on the same folds.
- Interpretation: RF impurity importance and Elastic Net coefficients.
- Coupling: machinery (12 genes) vs inflammation (18 genes) mean z-scores,
  Pearson r and age-partial r via residualisation.
- Robustness: 2,000-resample bootstrap CIs, performance by sex and age tertile,
  GTEx–OneK1K concordance, ROC/PR under three binarisations, stepwise
  Elastic Net → concordant panel → XGBoost → Ridge-stacked ensemble.

Random seeds are fixed at 42 for every stochastic step.
