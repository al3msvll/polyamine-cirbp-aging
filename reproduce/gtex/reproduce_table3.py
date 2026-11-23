"""Thesis Table 3: age association of the four sub-scores and the integrative PCAS
in GTEx v11 whole blood (n = 803).

Usage:
    python reproduce_table3.py <gene_tpm_v11_whole_blood.gct.gz> <SubjectPhenotypesDS.txt> [out_dir]
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from axis_score_definitions import PANEL, axis_scores

gct, pheno = sys.argv[1], sys.argv[2]
out = sys.argv[3] if len(sys.argv) > 3 else "."
genes = sorted({g for v in PANEL.values() for g in v})

ph = pd.read_csv(pheno, sep="\t").set_index("SUBJID")
tpm = pd.read_csv(gct, sep="\t", skiprows=2)
tpm = tpm[tpm["Description"].isin(genes)]
sym = tpm["Description"]
tpm = tpm.drop(columns=[c for c in ("Name", "Description", "id") if c in tpm.columns])
X = np.log1p(tpm.groupby(sym.values).max().T.astype(float))

subject = X.index.map(lambda s: "-".join(s.split("-")[:2]))
decade_mid = lambda a: (int(a[:2]) + int(a[3:5])) / 2
age = pd.Series([decade_mid(ph.loc[s, "AGE"]) if s in ph.index else np.nan for s in subject], index=X.index)
X, age = X[age.notna()], age.dropna()

scores, pca = axis_scores(X, age)
table = pd.DataFrame([(c, *stats.pearsonr(scores[c], age)) for c in scores.columns], columns=["score", "r", "p"])
print(f"n = {len(age)}")
print(table.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
print(f"PC1 variance explained: {100 * pca.explained_variance_ratio_[0]:.1f}%")
print("PC1 loadings:", {k: round(float(v), 2) for k, v in zip(scores.columns[:4], pca.components_[0])})
table.to_csv(os.path.join(out, "table3_subscores_pcas.csv"), index=False)
