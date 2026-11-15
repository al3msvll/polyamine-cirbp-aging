# Test run

python smoke_test.py --full, 2026-11-15, Python 3.13.18, 2 CPUs.
Synthetic inputs: GTEx n=120, GSE233321 n=60 (48 adults), OneK1K n=150. All stages executed; the numbers are meaningless.

## Tables
- gtex_nnls_cell_fractions.csv
- oof_GSE233321_Elastic_Net_panel.csv
- oof_GSE233321_Random_Forest_(tuned)_panel.csv
- oof_GSE233321_Random_Forest_panel.csv
- oof_GTEx_Elastic_Net_panel+cells+sex.csv
- oof_GTEx_Elastic_Net_panel.csv
- oof_GTEx_Random_Forest_panel.csv
- oof_OneK1K_Elastic_Net_panel+sex.csv
- oof_OneK1K_Random_Forest_panel+sex.csv
- scores_GSE233321.csv
- scores_GSE233321_full.csv
- scores_GTEx.csv
- scores_OneK1K.csv
- table01_cohorts.csv
- table02_GTEx_top20_age_genes.csv
- table03_GSE233321_full_score_age.csv
- table03_GSE233321_score_age.csv
- table03_GTEx_score_age.csv
- table03_OneK1K_score_age.csv
- table04_GSE233321_elastic_net_folds.csv
- table05_GSE233321_random_forest_folds.csv
- table06_GSE233321_rf_sweep.csv
- table06b_GSE233321_rf_grid.csv
- table07_GSE233321_rf_tuned_folds.csv
- table08_clock_summary.csv
- table09_GTEx_importances_top15.csv
- table10_coupling.csv
- table11_OneK1K_cell_fractions_age.csv
- table12_bootstrap_ci.csv
- table13_stratified_performance.csv
- table14_concordance_groups.csv
- table14b_concordance_genes.csv
- table15_group_effect_sizes.csv
- table17_GTEx_age_association.csv
- table18_OneK1K_age_association.csv
- table19_GSE233321_age_association.csv
- table19b_GSE233321_full_age_association.csv
- table20_GTEx_importances_all.csv
- table_roc_OneK1K.csv
- table_stepwise_OneK1K.csv

## Figures
- fig04_volcano_GTEx.png
- fig05_volcano_GSE233321.png
- fig06_volcano_OneK1K.png
- fig07_GTEx_PCAS_by_decade.png
- fig08_GSE233321_per_fold_metrics.png
- fig09_GSE233321_fold_r2.png
- fig10_rf_hyperparameter_ranges.png
- fig11_GTEx_clock_and_models.png
- fig12_GSE233321_summary.png
- fig13_OneK1K_summary.png
- fig14_GTEx_importances_top20.png
- fig15_coupling.png
- fig16_OneK1K_cell_fractions.png
- fig17_concordance.png
- fig18_OneK1K_roc_pr.png
- fig19_OneK1K_stepwise.png
- fig21_dataset_bias.png
