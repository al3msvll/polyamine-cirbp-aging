"""Thesis Table 9: GTEx Random Forest importances and Elastic Net coefficients
(full-data fits on standardised panel expression; RF 200 trees, max_depth 10,
min_samples_leaf 3, seed 42; ElasticNetCV l1 0.1-0.9, 40 alphas, cv 5).

Usage:
    python reproduce_table9.py <config.json> [out_dir]
"""
import os
import sys
import json
import warnings
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import tfm_pipeline as tp

cfg = json.load(open(sys.argv[1]))
out = sys.argv[2] if len(sys.argv) > 2 else "."
g = tp.load_gtex(cfg)
X, y = g.expr.T, g.age.values
Xs = StandardScaler().fit_transform(X)

rf = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=3,
                           random_state=42, n_jobs=-1).fit(Xs, y)
alphas = {"n_alphas": 40} if "n_alphas" in ElasticNetCV.__init__.__code__.co_varnames else {"alphas": 40}
en = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9], cv=5, max_iter=20000,
                  random_state=42, **alphas).fit(Xs, y)

table = (pd.DataFrame({"rf_importance": rf.feature_importances_, "en_coefficient": en.coef_}, index=X.columns)
         .sort_values("rf_importance", ascending=False).head(15).round(3))
print(table)
table.to_csv(os.path.join(out, "table9_importances.csv"))
