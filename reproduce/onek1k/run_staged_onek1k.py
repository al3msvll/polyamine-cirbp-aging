"""Staged model comparison on OneK1K (thesis Section 3.12 / 4.12-4.13).

Data: data/onek1k_logcpm.csv (981 donors, exact ages 19-97; donor-summed UMI,
log2-CPM). Features: 85 panel genes detected at CPM >= 1 in >= 20 % of donors.

Stages under one outer 10-fold KFold (shuffle, seed 42), per-fold standardisation:
  0 Elastic Net (ElasticNetCV, inner 5-fold, l1 0.1-0.9, 40 alphas)
  1 Elastic Net restricted to the genes FDR<0.05 in both GTEx and OneK1K (Annex C)
  2 XGBoost, RandomizedSearchCV (n_iter=60, inner 5-fold) per outer fold
  3 Ridge stacker on inner out-of-fold base predictions (fully nested)
Writes results/onek1k/staged_onek1k_results.json, per-fold and out-of-fold CSVs.
"""
import sys, json, warnings
warnings.filterwarnings('ignore')
import os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.environ.get('TFM_DATA', os.path.join(ROOT, 'data'))
OUT = os.environ.get('TFM_OUT', os.path.join(ROOT, 'results', 'onek1k'))
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV, Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import stats
from xgboost import XGBRegressor

log_expr = pd.read_csv(os.path.join(DATA, 'onek1k_logcpm.csv'), index_col=0)
meta = pd.read_csv(os.path.join(DATA, 'onek1k_meta.csv'), index_col=0).loc[log_expr.index]
y = meta['age'].values.astype(float)
cpm = np.exp2(log_expr) - 1
keep85 = log_expr.columns[((cpm >= 1).mean(axis=0) >= 0.20).values]
X85 = log_expr[keep85].values
print(f'matrix: {X85.shape[0]} donors x {len(keep85)} genes (85-gene detected set)')

# genes FDR<0.05 in both GTEx and OneK1K (thesis Annex C)
gtex = pd.read_csv(os.path.join(DATA, 'gtex_age_association.csv'))
onek = pd.read_csv(os.path.join(DATA, 'onek1k_age_association.csv'))
g_sig = set(gtex.loc[pd.to_numeric(gtex.FDR, errors='coerce') < 0.05, 'Gene'])
o_sig = set(onek.loc[pd.to_numeric(onek.FDR, errors='coerce') < 0.05, 'Gene'])
concordant = sorted(g_sig & o_sig)
conc_idx = [i for i, g in enumerate(keep85) if g in concordant]
missing = [g for g in concordant if g not in set(keep85)]
print(f'concordant set: {len(concordant)} genes; present in 85-gene matrix: {len(conc_idx)}; missing: {missing}')
Xc = X85[:, conc_idx]

kf = KFold(n_splits=10, shuffle=True, random_state=42)
L1 = [0.1, 0.3, 0.5, 0.7, 0.9]

def en_fit_predict(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    en = ElasticNetCV(l1_ratio=L1, n_alphas=40, cv=5, random_state=42,
                      max_iter=20000, n_jobs=-1)
    en.fit(sc.transform(Xtr), ytr)
    return en.predict(sc.transform(Xte))

XGB_GRID = {
    'n_estimators': [200, 400, 800],
    'max_depth': [2, 3, 4, 6],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.4, 0.6, 0.8, 1.0],
    'reg_alpha': [0.0, 0.1, 1.0],
    'reg_lambda': [1.0, 5.0, 10.0],
}

def xgb_fit_predict(Xtr, ytr, Xte, seed=42):
    sc = StandardScaler().fit(Xtr)
    xgb = XGBRegressor(random_state=seed, n_jobs=4, tree_method='hist')
    search = RandomizedSearchCV(xgb, XGB_GRID, n_iter=60, cv=5,
                                scoring='r2', random_state=seed, n_jobs=4)
    search.fit(sc.transform(Xtr), ytr)
    return search.best_estimator_.predict(sc.transform(Xte)), search.best_params_

stages = {'EN_full': X85, 'EN_concordant': Xc}
oof = {k: np.zeros(len(y)) for k in ['EN_full', 'EN_concordant', 'XGB', 'STACK']}
fold_rows = []
xgb_params = []

for fold, (tr, te) in enumerate(kf.split(X85)):
    # stage 0/1: elastic nets
    p_en = en_fit_predict(X85[tr], y[tr], X85[te])
    p_enc = en_fit_predict(Xc[tr], y[tr], Xc[te])
    # stage 2: xgboost with inner random search
    p_xgb, bp = xgb_fit_predict(X85[tr], y[tr], X85[te])
    xgb_params.append(bp)
    # stage 3: nested stacking — base OOF inside outer-train via inner 5-fold
    inner = KFold(n_splits=5, shuffle=True, random_state=42)
    base_oof = np.zeros((len(tr), 3))
    for itr, ite in inner.split(X85[tr]):
        base_oof[ite, 0] = en_fit_predict(X85[tr][itr], y[tr][itr], X85[tr][ite])
        base_oof[ite, 1] = en_fit_predict(Xc[tr][itr], y[tr][itr], Xc[tr][ite])
        base_oof[ite, 2], _ = xgb_fit_predict(X85[tr][itr], y[tr][itr], X85[tr][ite], seed=42)
    sc_meta = StandardScaler().fit(base_oof)
    ridge = Ridge(alpha=1.0)
    ridge.fit(sc_meta.transform(base_oof), y[tr])
    test_base = np.column_stack([p_en, p_enc, p_xgb])
    p_stack = ridge.predict(sc_meta.transform(test_base))

    oof['EN_full'][te] = p_en
    oof['EN_concordant'][te] = p_enc
    oof['XGB'][te] = p_xgb
    oof['STACK'][te] = p_stack
    for name, p in [('EN_full', p_en), ('EN_concordant', p_enc), ('XGB', p_xgb), ('STACK', p_stack)]:
        fold_rows.append({'fold': fold, 'stage': name, 'n_test': len(te),
                          'R2': r2_score(y[te], p), 'MAE': mean_absolute_error(y[te], p)})
    print(f'fold {fold}: ' + '  '.join(f'{n} R2={r2_score(y[te], p):.3f}'
          for n, p in [('EN', p_en), ('ENc', p_enc), ('XGB', p_xgb), ('STK', p_stack)]))

results = {}
for name, p in oof.items():
    r, _ = stats.pearsonr(y, p)
    results[name] = {'R2': float(r2_score(y, p)), 'MAE': float(mean_absolute_error(y, p)),
                     'pearson_r': float(r)}
    print(f"{name}: pooled OOF R2={results[name]['R2']:.3f} MAE={results[name]['MAE']:.2f} r={results[name]['pearson_r']:.3f}")

base = results['EN_full']['R2']
for name in results:
    results[name]['gain_vs_EN_pct'] = float((results[name]['R2'] - base) / abs(base) * 100)

pd.DataFrame(fold_rows).to_csv(os.path.join(OUT, 'staged_onek1k_perfold.csv'), index=False)
pd.DataFrame({'age': y, **{f'pred_{k}': v for k, v in oof.items()}}).to_csv(os.path.join(OUT, 'staged_onek1k_oof.csv'), index=False)
with open(os.path.join(OUT, 'staged_onek1k_results.json'), 'w') as f:
    json.dump({'results': results, 'n_donors': int(len(y)), 'n_genes_full': int(len(keep85)),
               'n_genes_concordant': int(len(conc_idx)),
               'concordant_genes': [g for g in concordant if g in set(keep85)],
               'xgb_best_params_per_fold': [{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                             for k, v in bp.items()} for bp in xgb_params]}, f, indent=2)
print('saved staged_onek1k_results.json / perfold / oof')
