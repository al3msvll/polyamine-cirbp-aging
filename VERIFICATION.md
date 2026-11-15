# Verification against the manuscript (15 Nov 2026)

The pipeline was run on the real public inputs and its output compared with
the values printed in the TFM. "Identical" means equal at the precision the
manuscript reports.

Inputs used

| Cohort | File | Size |
|---|---|---|
| GTEx v11 | `gene_tpm_v11_whole_blood.gct.gz` (74,628 genes × 803 samples) + `GTEx_Analysis_v11_Annotations_SubjectPhenotypesDS.txt` | 94 MB |
| LM22 | CIBERSORT LM22 (547 genes × 22 types) | 0.1 MB |
| GSE233321 | GEO `GSE233321_all_PBMCs.h5ad` (1,191,563 cells × 21,222 genes in `raw/X`, 167 samples, `Age_in_yrs`, no sex) | 12.5 GB |
| OneK1K | CELLxGENE `1e44db10-b572-46cc-adae-dcc7acd44ca6.h5ad` (1,248,980 cells × 35,528 genes, 981 donors, integer `age`) | 4.4 GB |

## Cohorts (Table 1)

| | Manuscript | Pipeline |
|---|---|---|
| GTEx donors / age range / mean | 803 / 25–75 / ~52 | 803 / 25–75 / 53.3 |
| GTEx sex | 358 F / 445 M (§4.1); 262 F / 541 M (Table 13) | 262 F / 541 M |
| GSE233321 adults / range / mean | 91 / 18–105 / 58.3 | 91 / 18–105 / 58.3 |
| OneK1K donors / range / mean / SD | 981 / 19–97 / 63.9 / 17.4 | 981 / 19–97 / 63.9 / 16.5 |
| OneK1K sex | 565 F / 416 M | 565 F / 416 M |
| Panel genes detected | 106 / 102 / 85 | 106 / 102 (MMP3, MMP10, MMP12, NPPB absent) / 85 (MMP10, MMP12 absent) |
| LM22 genes shared with GTEx | 515 | 515 |

## Gene-level age association (§4.2, §4.5, §4.6, Annex C)

| | Manuscript | Pipeline |
|---|---|---|
| GTEx genes at FDR<0.05 | 81/106 | 81/106 |
| GTEx median \|β\| | 0.14 | 0.147 |
| GTEx Table 2 (β, FDR): JUN, ATF3, VEGFA, CXCL8, MYD88, PSEN1, NR4A1, PLAUR, FOS, CDKN1A, TLR4, GADD45B | +0.368 (2.9e-25), +0.349, +0.350, +0.348, −0.322, −0.318, +0.313, +0.308, +0.303, +0.301, −0.298, +0.300 | identical, including FDR |
| GSE233321 adults / full lifespan at FDR<0.05 | 14/102 / 65/102 | 14/102 / 65/102 |
| OneK1K at FDR<0.05 | 58/85 | 58/85 |
| OneK1K Table 18 order | CDKN2A, IGFBP3, MYC, BAX, IGFBP7, HSPB1, TNF, FAS … | identical |
| CIRBP β GTEx / OneK1K | −0.115 / +0.144 | −0.115 / +0.145 |

## Sub-scores and PCAS (§4.3, §4.5, §4.6, Table 3)

| | Manuscript | Pipeline |
|---|---|---|
| GTEx hypusination r, p | +0.094, 0.00742 | +0.094, 0.00742 |
| GTEx PPAS / CIRBP module / SASP r | −0.009 / +0.130 / +0.161 (+0.155 in text) | −0.149 / +0.210 / +0.210 |
| GTEx PCAS r, p; PC1 variance | +0.128, 2.7e-4; 38.4% | +0.151, 1.7e-5; 65.9% |
| GSE233321 adults PPAS / hyp / CIRBP / SASP / PCAS | −0.030 / +0.131 / −0.023 / +0.095 / −0.023 (p 0.83); PC1 60% | −0.031 / +0.131 / +0.021 / +0.095 / −0.006 (p 0.96); PC1 59% |
| OneK1K PCAS r, p | +0.184, 6.8e-9 (~3.4% variance) | +0.185, 5.5e-9 (3.4%) |

The GTEx PCAS block (Table 3, Figure 7, PCAS rows of Table 12) is the only
part of the Results that does not reproduce from the definition in §3.6; the
same definition reproduces the GSE233321 and OneK1K sub-scores, so the
original GTEx sub-scores were built from a different gene set.

## Age clocks (§4.4–4.6, Tables 4–8)

| | Manuscript | Pipeline |
|---|---|---|
| GTEx Elastic Net, panel: R² / MAE / r | 0.218 / 9.22 / 0.468 | 0.215 / 9.28 / 0.465 |
| GTEx Random Forest, panel | 0.195 / 9.35 | 0.210 / 9.32 |
| GTEx Elastic Net, panel + 22 NNLS fractions + sex | 0.207 / 9.30 | 0.203 / 9.38 |
| GSE233321 Elastic Net | 0.567 / 13.50 / 0.753 | 0.577 / 13.37 / 0.760 |
| GSE233321 Random Forest, baseline | 0.488 / 15.2 / 0.765 | 0.500 / 14.92 / 0.761 |
| GSE233321 Random Forest, tuned | 0.555 / 13.9 / 0.785 | 0.555 / 13.80 / 0.771 |
| GSE233321 best RF configuration | 600 trees, depth 5, leaf 2, max_features 0.5 | 600 trees, depth 5, leaf 2, max_features 0.5 |
| GSE233321 most / least influential hyper-parameter | min_samples_leaf (range 0.188) / n_estimators (0.005) | min_samples_leaf (0.156) / n_estimators (0.018) |
| GSE233321 Elastic Net per-fold R² | −0.59 to 0.79, mean 0.435, SD 0.411, fold 3 negative | 0.22 to 0.88, mean 0.531, SD 0.201, no negative fold |
| OneK1K Elastic Net, panel + sex | 0.468 / 9.61 / 0.68 | 0.4685 / 9.607 / 0.685 |
| OneK1K Random Forest | 0.396 / 10.07 | 0.359 / 10.38 |
| OneK1K by sex: F (n=565) / M (n=416) | R² 0.50, MAE 9.4 / 0.42, 9.9 | 0.50, 9.39 / 0.42, 9.90 |
| OneK1K age tertiles (n=327/329/325) MAE | 12.1 / 6.3 / 10.5 | 12.0 / 6.4 / 10.4 |
| GTEx by sex: F (262) / M (541) | 0.23, 9.0 / 0.21, 9.3 | 0.23, 9.12 / 0.21, 9.36 |

Pooled clock metrics agree to ±0.01–0.04. The per-fold tables do not match
fold by fold because the manuscript's folds came from a different shuffle of
the donors; with nine test donors per fold the single-fold values are not
stable, which is the manuscript's own point in §4.4a.

## Model interpretation (§4.7, Table 9)

Both runs find a distributed signal with no gene above 5 % of RF importance,
led by JUN, VEGFA, NR4A1, ATF3, PLAUR, CXCL8 (manuscript order: NR4A1, JUN,
VEGFA, CXCL8, IGFBP3). The individual importance values differ (manuscript
max 0.051, pipeline 0.023); the manuscript's forest was evidently fitted
with different settings than the §3.7 baseline.

## Machinery–inflammation coupling (§4.8, Table 10, Table 12)

| | Manuscript | Pipeline |
|---|---|---|
| GTEx r, p, age-partial r | +0.628, ≈2e-89, +0.628 | +0.628, 2.4e-89, +0.628 |
| GTEx machinery~age, inflammation~age | +0.054, +0.149 | +0.054, +0.149 |
| OneK1K r, p, partial | +0.599, ≈1e-96, +0.594 | +0.599, 1.0e-96, +0.594 |
| OneK1K machinery~age, inflammation~age | +0.130, +0.100 | +0.130, +0.100 |
| GSE233321 adults r, p, partial | +0.198, 0.06, +0.200 | +0.198, 0.060, +0.200 |
| GSE233321 full r, p, partial | +0.414, 2.6e-8, +0.374 | +0.414, 2.7e-8, +0.374 |
| GSE233321 machinery~age, inflammation~age (adults / full) | +0.098 / +0.246, −0.008 / +0.262 | +0.098 / +0.246, −0.008 / +0.262 |
| Bootstrap CI, coupling GTEx / OneK1K | [+0.577, +0.678] / [+0.563, +0.635] | [+0.576, +0.676] / [+0.562, +0.634] |
| Bootstrap CI, PCAS~age OneK1K | [+0.126, +0.243] | [+0.123, +0.246] |

## Immune composition (§4.9, Table 11)

All 17 cell types with mean abundance >0.5 % reproduce with the same
fractions and correlations (naive CD8 T −0.680, MAIT −0.297, naive CD4 T
−0.200 … NK +0.308); 11 of 17 significant.

## Cross-cohort concordance (§4.11, Tables 14–15, Figure 17)

| | Manuscript | Pipeline |
|---|---|---|
| Shared genes / replicated / same direction | 85 / 46 / 31 (67.4 %) | 85 / 46 / 31 (67.4 %) |
| Pearson r, p; Spearman ρ, p on β | +0.144, 0.19; +0.166, 0.13 | +0.144, 0.189; +0.166, 0.129 |
| Table 14 counts per group | 24/14/6/4/0, 4/1/1/0/2, 15/10/1/3/1, 7/3/1/1/2, 28/15/7/2/4, 6/2/2/2/0 | identical |
| 15 discordant genes | incl. MYD88, TLR4, TLR2, NFKBIA, TRAF6, CIRBP, HMGB1, HSPB1, MMP9, HIF1A | the same 15: AMD1, CIRBP, FAS, HGF, HIF1A, HMGB1, HSPB1, IGFBP7, IL1B, MMP9, MYD88, NFKBIA, TLR2, TLR4, TRAF6 |
| Table 15 mean \|β\| per group, GTEx / OneK1K | 0.115/0.144, 0.185/0.126, 0.077/0.061, 0.159/–, 0.155/0.124, 0.101/0.104, 0.155/0.110, 0.167/0.127 | identical to ±0.001 |

The per-group "direction consistency" column of Table 14 (58/50/50/60/58/50 %)
is not reproduced by either definition (among replicated genes: 71/100/50/67/73/100 %;
among all shared genes: 58/50/53/71/54/50 %); the overall 67.4 % is.

## ROC and stepwise model improvement (§4.12–4.13, Figures 18–19)

The manuscript ran this stage on a re-extracted OneK1K matrix (Census snapshot
2025-11-08; 42 of the 46 concordant genes present), so its baseline (0.496)
differs from its own §4.6 value; the pipeline ran it on the original matrix
(baseline 0.469, all 46 concordant genes present).

| | Manuscript | Pipeline |
|---|---|---|
| Elastic Net full → concordant → XGBoost → stacked, R² | 0.496 → 0.432 (−12.9 %) → 0.471 (−5.1 %) → 0.513 (+3.3 %) | 0.469 → 0.433 (−7.7 %) → 0.453 (−3.4 %) → 0.486 (+3.6 %) |
| MAE, Elastic Net → stacked | 9.35 → 9.18 (−0.17) | 9.61 → 9.40 (−0.21) |
| AUC, extreme contrast (<45 vs ≥70) | 0.948–0.971 | 0.942–0.964 |
| AUC, clinical decade (<55 vs ≥65) | 0.887–0.909 | 0.876–0.902 |
| AUC, median split | 0.804–0.832 | 0.800–0.819 |
| Mean AUC, Elastic Net vs stacked | 0.903 vs 0.903 | 0.895 vs 0.893 |

Same ordering and same conclusions: the concordance filter and XGBoost alone
both fall below the Elastic Net, the stacked ensemble gives a small real gain
in regression, and none of it changes age-group discrimination.

## Not reproduced here

- The discarded 500FG/GSE134080 cohort (R² 0.181, coupling r −0.41, Figure 21
  panels C/F overlay) is not part of the pipeline.
- Figures 1–3 and 20 are drawn by hand or from the literature.
