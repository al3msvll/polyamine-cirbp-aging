"""Re-extract OneK1K from CZ CELLxGENE Census (snapshot 2025-11-08; dataset
3faad104-2ab8-4434-816d-474d8d2641db) as described in thesis Section 3.4a:
raw UMI summed per donor and panel gene; log2-CPM on each donor's total UMI.
Then: detection count, OLS expr ~ age + sex with BH-FDR, and the Elastic Net clock
(10-fold KFold, seed 42, per-fold standardisation; ElasticNetCV, inner 5-fold,
l1 0.1-0.9, 40 alphas). Expected: 85 genes, 58 FDR<0.05 (52 up / 6 down),
R2 0.468, MAE 9.61 y, coupling r 0.599.
Writes data/onek1k_logcpm.csv and data/onek1k_meta.csv.
"""
import sys, re, warnings
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
from scipy import sparse
import cellxgene_census
import tiledbsoma
from panel import UNIQUE_GENES

ONEK1K_ID = '3faad104-2ab8-4434-816d-474d8d2641db'
print('Opening Census 2025-11-08...')
census = cellxgene_census.open_soma(census_version='2025-11-08')

obs_filter = f"dataset_id == '{ONEK1K_ID}'"
print('Fetching obs (donor_id, development_stage, sex, raw_sum)...')
obs = cellxgene_census.get_obs(
    census, 'Homo sapiens',
    value_filter=obs_filter,
    column_names=['donor_id', 'development_stage', 'sex', 'raw_sum'],
)
print(f'cells: {len(obs)}')
print('obs columns:', list(obs.columns))

def parse_age(ds):
    if pd.isna(ds):
        return np.nan
    m = re.search(r'(\d+)-year-old', str(ds))
    return float(m.group(1)) if m else np.nan

obs['age'] = obs['development_stage'].apply(parse_age)
donor_lib = obs.groupby('donor_id')['raw_sum'].sum()
donor_age = obs.groupby('donor_id')['age'].first()
donor_sex = obs.groupby('donor_id')['sex'].first()
print(f'donors: {len(donor_lib)}, ages valid: {donor_age.notna().sum()}, '
      f'age {donor_age.min():.0f}-{donor_age.max():.0f}')
print('sex values:', donor_sex.value_counts().to_dict())

print('Fetching panel-gene expression (var-filtered query)...')
adata = cellxgene_census.get_anndata(
    census, 'Homo sapiens',
    obs_value_filter=obs_filter,
    var_value_filter=f"feature_name in {list(UNIQUE_GENES)}",
    obs_column_names=['donor_id'],
)
print(f'adata: {adata.shape}  (cells x panel genes found)')
gene_syms = list(adata.var['feature_name'].values)
print(f'panel genes present in dataset: {len(gene_syms)}')
missing = sorted(set(UNIQUE_GENES) - set(gene_syms))
print(f'panel genes absent: {len(missing)}: {missing}')

X = adata.X
if sparse.issparse(X):
    X = X.tocsr()
donors = adata.obs['donor_id'].astype(str).values
udonors, inv = np.unique(donors, return_inverse=True)
print('Summing counts per donor...')
pb = np.zeros((len(udonors), X.shape[1]), dtype=np.float64)
for j in range(X.shape[1]):
    col = X[:, j].toarray().ravel() if sparse.issparse(X) else np.asarray(X[:, j]).ravel()
    np.add.at(pb[:, j], inv, col)
pb = pd.DataFrame(pb, index=udonors, columns=gene_syms)
census.close()

# keep donors with valid age
valid = donor_age.notna()
donors_valid = donor_age[valid].index
pb = pb.loc[pb.index.isin(donors_valid)]
lib = donor_lib.loc[pb.index]
age = donor_age.loc[pb.index]
sex = donor_sex.loc[pb.index]
print(f'pseudobulk: {pb.shape} donors x genes')

cpm = pb.div(lib, axis=0) * 1e6
log_expr = np.log2(cpm + 1)
log_expr.to_csv(os.path.join(DATA, 'onek1k_logcpm.csv'))
pd.DataFrame({'age': age, 'sex': sex, 'lib': lib}).to_csv(os.path.join(DATA, 'onek1k_meta.csv'))
print('saved onek1k_logcpm.csv and onek1k_meta.csv to', DATA)

# ---- detection counts under several definitions ----
print('\nDetection definitions:')
print('  nonzero in >0% donors:', (pb > 0).any(axis=0).sum())
print('  nonzero in >=50% donors:', ((pb > 0).mean(axis=0) >= 0.5).sum())
print('  mean CPM >= 1:', (cpm.mean(axis=0) >= 1).sum())
print('  CPM>=1 in >=10% donors:', ((cpm >= 1).mean(axis=0) >= 0.10).sum())
print('  CPM>=1 in >=25% donors:', ((cpm >= 1).mean(axis=0) >= 0.25).sum())
print('  CPM>=1 in >=50% donors:', ((cpm >= 1).mean(axis=0) >= 0.50).sum())
