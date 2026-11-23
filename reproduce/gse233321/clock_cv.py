"""
clock_cv.py — leak-free 10-fold cross-validation for the age clocks.

Implements the protocol described in thesis Sections 3.7 / 3.7a:
  - donors partitioned once into 10 equal folds (KFold, seed 42)
  - z-standardisation fit on each fold's training donors only
  - Elastic Net (ElasticNetCV, 5-fold internal CV, l1_ratio grid,
    40-point log alpha grid) and Random Forest (baseline: 300 trees,
    max_features='sqrt') as the two model families
  - per-fold R2 / MAE / Pearson r plus pooled out-of-fold metrics
  - one-at-a-time RF hyperparameter sweep + focused grid search
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

SEED = 42

L1_RATIO_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]
N_ALPHAS = 40

# RF baseline (thesis 3.7: 300 trees, max_features='sqrt', otherwise defaults)
RF_BASELINE = dict(
    n_estimators=300,
    max_features="sqrt",
    random_state=SEED,
    n_jobs=-1,
)

# RF sweep grid (thesis 3.7a)
RF_SWEEP_GRID = {
    "n_estimators": [100, 300, 600, 1000],
    "max_depth": [3, 5, 10, 20, None],
    "min_samples_leaf": [1, 2, 5, 10],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2", 0.3, 0.5, 1.0],
    "bootstrap": [True, False],
}


def en_factory():
    """Elastic Net with internal 5-fold CV selecting l1_ratio + alpha."""
    return ElasticNetCV(
        l1_ratio=L1_RATIO_GRID,
        n_alphas=N_ALPHAS,
        cv=5,
        random_state=SEED,
        n_jobs=-1,
    )


def rf_factory(**overrides):
    """Random Forest; baseline configuration unless overridden."""
    params = dict(RF_BASELINE)
    params.update(overrides)
    return RandomForestRegressor(**params)


def _metrics(y_true, y_pred):
    """Pooled metrics on out-of-fold predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    mae = float(np.mean(np.abs(y_true - y_pred)))
    r, p = stats.pearsonr(y_pred, y_true)
    return {"R2_pooled": r2, "MAE_pooled": mae, "r_pooled": float(r), "r_p_pooled": float(p)}


def per_fold_cv(X, y, model_or_factory, n_splits=10, seed=SEED):
    """
    10-fold CV with a single shared partition (KFold seed 42).

    X: donors x features; y: donor ages.
    model_or_factory: estimator instance or zero-arg factory returning a
    fresh estimator (factories guarantee a clean fit per fold).

    Returns (folds_df, agg_dict, oof_predictions).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof = np.full(len(y), np.nan)
    rows = []
    callable_factory = callable(model_or_factory) and not hasattr(model_or_factory, "predict")
    for fold_i, (tr, te) in enumerate(kf.split(X), start=1):
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
        model = model_or_factory() if callable_factory else model_or_factory
        model.fit(Xtr, y[tr])
        pred = model.predict(Xte)
        oof[te] = pred

        r2 = 1.0 - np.sum((y[te] - pred) ** 2) / np.sum((y[te] - y[te].mean()) ** 2)
        mae = float(np.mean(np.abs(pred - y[te])))
        r = float(stats.pearsonr(pred, y[te])[0]) if len(te) > 2 else np.nan
        rows.append({
            "fold": fold_i,
            "n_train": len(tr),
            "n_test": len(te),
            "R2": float(r2),
            "MAE": mae,
            "r": r,
        })

    folds = pd.DataFrame(rows)
    agg = _metrics(y, oof)
    return folds, agg, oof


def rf_hyperparameter_sweep(X, y, n_splits=10, seed=SEED):
    """
    Stage 1: one-at-a-time sweep around the baseline (thesis Tab 6).
    Stage 2: focused grid over the most influential parameters.

    Returns (sweep_df, grid_df, best_dict).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    sweep_rows = []
    for param, values in RF_SWEEP_GRID.items():
        for v in values:
            params = {param: v}
            _, agg, _ = per_fold_cv(X, y, rf_factory(**params), n_splits=n_splits, seed=seed)
            sweep_rows.append({
                "sweep": param, "value": v,
                "R2": agg["R2_pooled"], "MAE": agg["MAE_pooled"], "r": agg["r_pooled"],
            })
    sweep_df = pd.DataFrame(sweep_rows)

    # Stage 2: focused grid on the three most influential parameters
    spans = {
        p: float(sweep_df[sweep_df["sweep"] == p]["R2"].max()
                 - sweep_df[sweep_df["sweep"] == p]["R2"].min())
        for p in sweep_df["sweep"].unique()
    }
    top = sorted(spans, key=lambda k: -spans[k])[:3]
    focused_params = {p: RF_SWEEP_GRID[p] for p in top}

    import itertools
    grid_rows = []
    keys = list(focused_params)
    best = None
    for combo in itertools.product(*[focused_params[k] for k in keys]):
        params = dict(zip(keys, combo))
        if params.get("min_samples_leaf", 1) > params.get("min_samples_split", 2):
            continue  # skip invalid sklearn combinations
        _, agg, _ = per_fold_cv(X, y, rf_factory(**params), n_splits=n_splits, seed=seed)
        row = {"R2": agg["R2_pooled"], "MAE": agg["MAE_pooled"], "r": agg["r_pooled"], **params}
        grid_rows.append(row)
        if best is None or row["R2"] > best["R2"]:
            best = row
    grid_df = pd.DataFrame(grid_rows)
    return sweep_df, grid_df, best


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(91, 100))
    y = rng.uniform(18, 105, size=91)
    folds, agg, oof = per_fold_cv(X, y, en_factory())
    print(f"smoke EN R2={agg['R2_pooled']:.3f}")
    folds, agg, oof = per_fold_cv(X, y, rf_factory())
    print(f"smoke RF R2={agg['R2_pooled']:.3f}")
