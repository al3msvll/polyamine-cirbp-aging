"""Thesis Tables 4-7: GSE233321 adult age clock (n = 91), per-fold 10-fold cross-validation.

Input   data/GSE233321_pseudobulk_counts_rawX.parquet and data/GSE233321_donor_metadata.csv
        (make_pseudobulk_rawX.py); log2-CPM on the all-gene library size; adults >= 18 y.
Panel   clock_panel.py (103 genes; 97 detected in GSE233321).
Models  clock_cv.py: KFold(10, shuffle, seed 42), per-fold standardisation;
        Table 4 Elastic Net, Table 5 Random Forest (300 trees, sqrt),
        Table 6 one-at-a-time RF sweep + grid over the three most influential parameters,
        Table 7 tuned RF (600 trees, max_depth 5, min_samples_leaf 2, max_features 0.5).
Env     Python 3.11, scikit-learn 1.3.2. Runtime ~15 min on 2 cores.

Usage:
    python reproduce_tables_4_to_7.py [data_dir] [out_dir]
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from clock_cv import per_fold_cv, en_factory, rf_factory, rf_hyperparameter_sweep
from clock_panel import UNIQUE_GENES

warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "results", "gse233321")
os.makedirs(OUT, exist_ok=True)
REFERENCE = {"table4_elastic_net": "table4_elastic_net.csv",
             "table5_random_forest": "table5_random_forest.csv",
             "table7_random_forest_tuned": "table7_random_forest_tuned.csv"}


def load():
    counts = pd.read_parquet(os.path.join(DATA, "GSE233321_pseudobulk_counts_rawX.parquet"))
    expr = np.log2(counts.div(counts.sum(axis=0), axis=1) * 1e6 + 1)
    meta = pd.read_csv(os.path.join(DATA, "GSE233321_donor_metadata.csv"), index_col=0).loc[expr.columns]
    adults = meta.index[meta["age"] >= 18]
    genes = [g for g in UNIQUE_GENES if g in expr.index]
    print(f"{len(genes)}/{len(UNIQUE_GENES)} panel genes, {len(meta)} donors, {len(adults)} adults")
    return expr.loc[genes, adults].T.values, meta.loc[adults, "age"].values


def report(name, folds, agg, comparison):
    folds.to_csv(os.path.join(OUT, f"{name}_per_fold.csv"), index=False)
    ref = pd.read_csv(os.path.join(HERE, "reference", REFERENCE[name])).iloc[:, 3].values
    for i, (thesis, rerun) in enumerate(zip(ref, folds["R2"]), start=1):
        comparison.append({"table": name, "fold": i, "thesis_R2": thesis,
                           "rerun_R2": round(rerun, 3), "difference": round(rerun - thesis, 3)})
    print(f"{name}: pooled R2 {agg['R2_pooled']:.3f}, MAE {agg['MAE_pooled']:.2f} y")
    print("  rerun  ", folds["R2"].round(3).tolist())
    print("  thesis ", ref.tolist())


def main():
    X, y = load()
    comparison, summary = [], {}

    for name, model in [("table4_elastic_net", en_factory()), ("table5_random_forest", rf_factory())]:
        folds, agg, _ = per_fold_cv(X, y, model)
        report(name, folds, agg, comparison)
        summary[name] = agg

    sweep, grid, best = rf_hyperparameter_sweep(X, y)
    sweep.to_csv(os.path.join(OUT, "table6_random_forest_sweep.csv"), index=False)
    grid.to_csv(os.path.join(OUT, "table6_random_forest_grid.csv"), index=False)
    spans = {p: round(float(sweep[sweep.sweep == p].R2.max() - sweep[sweep.sweep == p].R2.min()), 3)
             for p in sweep.sweep.unique()}
    params = {k: best[k] for k in ("max_depth", "min_samples_leaf", "min_samples_split", "max_features", "bootstrap")
              if k in best and pd.notna(best[k])}
    for k in ("max_depth", "min_samples_leaf", "min_samples_split"):
        if k in params:
            params[k] = int(params[k])
    params["n_estimators"] = 600
    print("table6: R2 span per parameter", spans, "| selected", params)

    folds, agg, _ = per_fold_cv(X, y, rf_factory(**params))
    report("table7_random_forest_tuned", folds, agg, comparison)
    summary.update(table7_random_forest_tuned=agg, table6_spans=spans,
                   table7_parameters={k: str(v) for k, v in params.items()})

    pd.DataFrame(comparison).to_csv(os.path.join(OUT, "comparison_thesis_vs_rerun.csv"), index=False)
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
