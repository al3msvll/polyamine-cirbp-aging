#!/usr/bin/env python3
"""
El eje poliamina-CIRBP y el inflammaging / The Polyamine-CIRBP Axis and Inflammaging
Analysis pipeline for the Master's thesis (UNIR, MSc Bioinformatics, 2025-2026).
Author: Alejandra Martín Sevilla

Implements Methods 3.2-3.9 of the thesis and regenerates the tables and figures of
Results 4.1-4.13 from the public inputs listed in data/README.md.

    python tfm_pipeline.py --config config.json --out results

Cohorts whose input files are absent from the config are skipped, so the GTEx
part can be run on its own. Numbering of tables/figures in the output folder
follows the thesis.
"""

import argparse
import json
import gzip
import os
import sys
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import h5py
from scipy import sparse, stats
from scipy.optimize import nnls
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_predict
from sklearn.linear_model import ElasticNetCV, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (r2_score, mean_absolute_error, roc_auc_score,
                             average_precision_score, roc_curve,
                             precision_recall_curve)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SEED = 42
N_FOLDS = 10
N_BOOT = 2000
FDR_ALPHA = 0.05


# ---------------------------------------------------------------------------
# 3.3  Gene panel (Annex A, Table 16)
# ---------------------------------------------------------------------------
PANEL_GROUPS = OrderedDict([
    ("polyamine", ["ODC1", "SRM", "SMS", "AMD1", "SAT1", "SMOX", "PAOX"]),
    ("hypusination_machinery", ["EIF5A", "EIF5A2", "DHPS", "DOHH"]),
    ("hypusination_polyproline_targets", [
        "ATXN1", "ATXN2", "ATXN3", "ATXN7", "FMR1", "MEF2A", "MEF2C", "CALR",
        "TIMP1", "PSEN1", "PCNA", "BCL2L11", "CDK1", "CDK6", "DRAM2", "EIF4G2",
        "FOXM1", "MAP2K3", "PSMD7", "RAB7A"]),
    ("cirbp_axis_core", ["CIRBP"]),
    ("cirbp_direct_targets", [
        "TNF", "IL6", "VEGFA", "HIF1A", "HMGB1", "CCND1", "BCL2", "BAX",
        "GADD45B", "MYC", "CCNB1", "CST3", "MMP9", "NFE2L2", "XIAP", "BIRC5",
        "E2F1", "RPS3", "HSPA1A", "HSPB1", "ATF3", "DUSP1", "FOS", "JUN",
        "EGR1", "NR4A1", "CDKN1A", "MDM2"]),
    ("sasp_signature", [
        "IL6", "CXCL8", "IL1A", "IL1B", "TNF", "CCL2", "CCL20", "CXCL1",
        "CXCL2", "CXCL10", "MMP1", "MMP3", "MMP10", "MMP12", "TIMP1", "IGFBP3",
        "IGFBP7", "VEGFA", "HGF", "PLAUR", "SERPINE1", "SERPINB2", "ICAM1",
        "FAS", "FASLG", "ANGPTL4", "PTGS2", "NPPB", "NRG1", "LTA"]),
    ("nfkb_targets", [
        "NFKB1", "NFKB2", "RELA", "RELB", "REL", "NFKBIA", "NFKBIB", "TLR2",
        "TLR4", "MYD88", "IRAK1", "IRAK4", "TRAF6", "TNFAIP3", "BIRC3"]),
    ("senescence_markers", ["CDKN1A", "CDKN2A", "TP53", "MKI67", "LMNB1", "RB1"]),
])

# Genes listed in more than one group are assigned to the last group in which
# they appear; this reproduces the group labels of Annex C (Tables 17-19).
GENE_GROUP = OrderedDict()
for _grp, _genes in PANEL_GROUPS.items():
    for _g in _genes:
        GENE_GROUP[_g] = _grp
PANEL_106 = list(GENE_GROUP)                                   # 106 unique genes
PANEL_ONEK1K = [g for g in PANEL_106
                if GENE_GROUP[g] != "hypusination_polyproline_targets"]  # 87 genes

# Sub-score gene sets (Methods 3.6)
POLYAMINE_BIOSYNTHESIS = ["ODC1", "SRM", "SMS", "AMD1"]
POLYAMINE_CATABOLISM = ["SAT1", "SMOX", "PAOX"]
HYPUSINATION_SET = ["EIF5A", "EIF5A2", "DHPS", "DOHH"]
CIRBP_MODULE_SET = PANEL_GROUPS["cirbp_axis_core"] + PANEL_GROUPS["cirbp_direct_targets"]
SASP_SET = PANEL_GROUPS["sasp_signature"]

# Non-overlapping sets for the inflammation analysis (Methods 3.9)
MACHINERY_SET = ["ODC1", "SRM", "SMS", "AMD1", "SAT1", "SMOX", "PAOX",
                 "EIF5A", "EIF5A2", "DHPS", "DOHH", "CIRBP"]
INFLAMMATION_SET = ["IL6", "TNF", "IL1A", "IL1B", "CXCL8", "CCL2", "CCL20",
                    "CXCL1", "CXCL2", "CXCL10", "PTGS2", "SERPINE1", "ICAM1",
                    "NFKB1", "NFKB2", "TNFAIP3", "TLR2", "TLR4"]
assert not set(MACHINERY_SET) & set(INFLAMMATION_SET)

# Random Forest settings (Methods 3.7, 3.7a)
RF_BASE = dict(n_estimators=300, max_features="sqrt", max_depth=None,
               min_samples_leaf=1, min_samples_split=2, bootstrap=True)
RF_SWEEP = OrderedDict([
    ("n_estimators", [100, 300, 600, 1000]),
    ("max_depth", [3, 5, 10, 20, None]),
    ("min_samples_leaf", [1, 2, 5, 10]),
    ("min_samples_split", [2, 5, 10]),
    ("max_features", ["sqrt", "log2", 0.3, 0.5, 1.0]),
    ("bootstrap", [True, False]),
])
RF_GRID = dict(n_estimators=[600], max_depth=[3, 5, 10, None],
               min_samples_leaf=[1, 2, 5], max_features=["sqrt", 0.3, 0.5, 1.0])

# Age-group binarisations for the ROC analysis (Results 4.12)
ROC_SPLITS = OrderedDict([
    ("extreme (<45 vs >=70)", (45, 70)),
    ("clinical decade (<55 vs >=65)", (55, 65)),
    ("median split", None),
])


class Cohort:
    """One donor x gene expression matrix with its metadata.

    expr   : log-scale expression, genes x samples, panel genes only
    linear : linear-scale expression (TPM/CPM) restricted to LM22 genes, or None
    meta   : samples x [age, sex_male (optional), donor]
    """

    def __init__(self, name, expr, meta, linear=None, panel=PANEL_106,
                 tissue="", assay="", age_label=""):
        common = meta.index.intersection(expr.columns)
        self.name = name
        present = expr.loc[[g for g in panel if g in expr.index], common]
        self.expr = present[(present != 0).any(axis=1)]      # a panel gene counts as detected
        self.undetected = sorted(set(panel) - set(self.expr.index))  # only when expressed
        self.meta = meta.loc[common]
        self.linear = None if linear is None else linear.loc[:, common]
        self.panel = list(panel)
        self.cell_fractions = None
        self.tissue, self.assay, self.age_label = tissue, assay, age_label

    @property
    def genes(self):
        return list(self.expr.index)

    @property
    def age(self):
        return self.meta["age"].astype(float)

    @property
    def has_sex(self):
        return "sex_male" in self.meta.columns

    def subset(self, mask, name):
        c = Cohort.__new__(Cohort)
        c.__dict__.update(self.__dict__)
        c.name = name
        c.meta = self.meta.loc[mask]
        c.expr = self.expr.loc[:, c.meta.index]
        c.linear = None if self.linear is None else self.linear.loc[:, c.meta.index]
        c.cell_fractions = (None if self.cell_fractions is None
                            else self.cell_fractions.loc[c.meta.index])
        return c


# ---------------------------------------------------------------------------
# 3.2  Data sources
# ---------------------------------------------------------------------------
def strip_version(ensembl_id):
    return ensembl_id.split(".")[0]


def collapse_duplicate_symbols(df):
    """Keep, for every duplicated symbol, the row with the highest mean expression."""
    if not df.index.duplicated().any():
        return df
    keep = np.zeros(len(df), dtype=bool)
    positions = {}
    for pos, (sym, mean) in enumerate(zip(df.index, df.mean(axis=1))):
        if sym not in positions or mean > positions[sym][1]:
            positions[sym] = (pos, mean)
    for pos, _ in positions.values():
        keep[pos] = True
    return df.iloc[keep]


def read_gct(path, keep_symbols):
    """Stream a GTEx .gct(.gz); return (symbol x sample TPM for keep_symbols,
    Ensembl(no version) -> symbol map for every gene in the file)."""
    keep = set(keep_symbols)
    rows, index, ens2sym = [], [], {}
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        fh.readline(); fh.readline()
        header = fh.readline().rstrip("\n").split("\t")
        samples = header[2:]
        for line in fh:
            ens, sym, rest = line.rstrip("\n").split("\t", 2)
            ens2sym[strip_version(ens)] = sym
            if sym in keep:
                rows.append(np.array(rest.split("\t"), dtype=float))
                index.append(sym)
    df = pd.DataFrame(np.vstack(rows), index=index, columns=samples)
    return collapse_duplicate_symbols(df), ens2sym


def read_gtex_phenotypes(path):
    """GTEx SubjectPhenotypesDS.txt (SUBJID, SEX 1=male 2=female, AGE '20-29')
    or a pre-parsed table with SUBJID, AGE_MID, SEX."""
    ph = pd.read_csv(path, sep=None, engine="python")
    ph.columns = [c.upper() for c in ph.columns]
    out = pd.DataFrame({"donor": ph["SUBJID"].astype(str)})
    if "AGE_MID" in ph.columns:
        out["age"] = ph["AGE_MID"].astype(float)
        out["age_bin"] = ph["AGE_BIN"] if "AGE_BIN" in ph.columns else np.nan
    else:
        bins = ph["AGE"].astype(str)
        lo = bins.str.extract(r"(\d+)")[0].astype(float)
        hi = bins.str.extract(r"-(\d+)")[0].astype(float)
        out["age"] = (lo + hi + 1) / 2           # 20-29 -> 25 (Table 1: 25-75)
        out["age_bin"] = bins.values
    sex = ph["SEX"].astype(str).str.upper().str.strip()
    out["sex_male"] = sex.isin(["1", "M", "MALE"]).astype(int).values
    return out.set_index("donor")


def load_gtex(cfg):
    lm22 = read_lm22(cfg["lm22"]) if cfg.get("lm22") else None
    wanted = set(PANEL_106) | (set(lm22.index) if lm22 is not None else set())
    tpm, ens2sym = read_gct(cfg["gtex_gct"], wanted)
    ph = read_gtex_phenotypes(cfg["gtex_phenotypes"])
    donors = pd.Series(["-".join(s.split("-")[:2]) for s in tpm.columns], index=tpm.columns)
    meta = ph.reindex(donors.values)
    meta.index = tpm.columns
    meta["donor"] = donors.values
    meta = meta.dropna(subset=["age"])
    expr = np.log1p(tpm.loc[[g for g in PANEL_106 if g in tpm.index]])   # log1p-TPM (3.4)
    linear = tpm.loc[tpm.index.intersection(lm22.index)] if lm22 is not None else None
    c = Cohort("GTEx", expr, meta, linear=linear, tissue="Whole blood",
               assay="RNA-seq (TPM)", age_label="decade bins")
    c.ens2sym = ens2sym
    return c


def read_lm22(path):
    lm22 = pd.read_csv(path, sep="\t", index_col=0)
    lm22 = lm22.loc[:, [c for c in lm22.columns if not c.startswith("Unnamed")]]
    return lm22.groupby(level=0).mean()


# ---------------------------------------------------------------------------
# 3.4a  Single-cell pseudobulk (GSE233321, OneK1K)
# ---------------------------------------------------------------------------
def _decode(arr):
    arr = np.asarray(arr)
    if arr.dtype.kind in ("S", "O"):
        return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr])
    return arr


def _read_column(f, group, key):
    """Read an obs/var column written by anndata: categorical (categories/codes),
    nullable array (values/mask) or plain dataset."""
    obj = f[group][key]
    if isinstance(obj, h5py.Group):
        if "codes" in obj:
            cats = _decode(obj["categories"][()])
            codes = obj["codes"][()]
            out = np.full(len(codes), None, dtype=object)
            out[codes >= 0] = cats[codes[codes >= 0]]
            return out
        values = _decode(obj["values"][()])
        if "mask" in obj:
            values = np.where(obj["mask"][()], None, values)
        return values
    values = obj[()]
    if "categories" in obj.attrs:                       # anndata < 0.8 layout
        cats = _decode(f[obj.attrs["categories"]][()])
        return cats[values]
    return _decode(values)


def _index_column(f, group):
    key = f[group].attrs.get("_index", "_index")
    key = key.decode() if isinstance(key, bytes) else str(key)
    return _read_column(f, group, key)


def _pick(columns, candidates, explicit=None):
    if explicit:
        if explicit not in columns:
            raise KeyError(f"obs column '{explicit}' not found; available: {sorted(columns)}")
        return explicit
    for c in candidates:
        if c in columns:
            return c
    return None


def _parse_age(values):
    """Numeric ages, or ages embedded in strings such as '63-year-old human stage'."""
    s = pd.Series(values)
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.9:
        return num.values
    return s.astype(str).str.extract(r"(\d+\.?\d*)")[0].astype(float).values


def pseudobulk_h5ad(path, donor_col=None, age_col=None, sex_col=None,
                    cell_type_col=None, chunk_cells=20000, ens2sym=None):
    """Sum raw counts per donor across all cells (Methods 3.4a).

    Returns counts (donor x gene, all genes), meta (donor x [age, sex_male]),
    and cell-type fractions per donor (or None). Uses raw/X when present.
    """
    with h5py.File(path, "r") as f:
        obs_cols = set(f["obs"].keys())
        donor_col = _pick(obs_cols, ["donor_id", "donor", "individual", "subject",
                                     "sample_id", "sample", "Donor"], donor_col)
        age_col = _pick(obs_cols, ["age", "Age", "age_years", "Age_in_yrs", "donor_age",
                                   "development_stage"], age_col)
        sex_col = _pick(obs_cols, ["sex", "Sex", "gender"], sex_col)
        cell_type_col = _pick(obs_cols, ["cell_type", "celltype", "predicted.celltype.l2"],
                              cell_type_col)
        if donor_col is None or age_col is None:
            raise KeyError(f"could not identify donor/age columns in {path}: {sorted(obs_cols)}")

        donors = _read_column(f, "obs", donor_col).astype(str)
        ages = _parse_age(_read_column(f, "obs", age_col))
        sexes = _read_column(f, "obs", sex_col).astype(str) if sex_col else None
        ctypes = _read_column(f, "obs", cell_type_col).astype(str) if cell_type_col else None

        use_raw = "raw" in f and "X" in f["raw"]
        matrix = f["raw"]["X"] if use_raw else f["X"]
        var_group = "raw/var" if (use_raw and "var" in f["raw"]) else "var"
        var_cols = set(f[var_group].keys())
        var_index = _index_column(f, var_group)
        symbol_col = next((c for c in ("feature_name", "gene_symbol", "symbol") if c in var_cols), None)
        if symbol_col:
            symbols = _read_column(f, var_group, symbol_col).astype(str)
        elif ens2sym is not None and var_index[0].startswith("ENSG"):
            symbols = np.array([ens2sym.get(strip_version(v), v) for v in var_index])
        else:
            symbols = var_index
        donor_codes, donor_names = pd.factorize(donors)
        n_donors, n_cells, n_genes = len(donor_names), len(donors), len(symbols)
        counts = np.zeros((n_donors, n_genes), dtype=np.float64)

        if isinstance(matrix, h5py.Group):
            enc = matrix.attrs.get("encoding-type", b"csr_matrix")
            enc = enc.decode() if isinstance(enc, bytes) else enc
            if enc != "csr_matrix":
                raise ValueError(f"{path}: expected csr_matrix, found {enc}")
            indptr = matrix["indptr"][()]
            for start in range(0, n_cells, chunk_cells):
                stop = min(start + chunk_cells, n_cells)
                a, b = indptr[start], indptr[stop]
                block = sparse.csr_matrix(
                    (matrix["data"][a:b], matrix["indices"][a:b], indptr[start:stop + 1] - a),
                    shape=(stop - start, n_genes))
                agg = sparse.csr_matrix(
                    (np.ones(stop - start), (donor_codes[start:stop], np.arange(stop - start))),
                    shape=(n_donors, stop - start))
                counts += (agg @ block).toarray()
        else:
            for start in range(0, n_cells, chunk_cells):
                stop = min(start + chunk_cells, n_cells)
                block = np.asarray(matrix[start:stop])
                np.add.at(counts, donor_codes[start:stop], block)

    if not np.allclose(counts[:5], np.round(counts[:5])):
        print(f"  warning: {path} matrix does not look like raw integer counts", file=sys.stderr)

    counts = pd.DataFrame(counts.T, index=symbols, columns=donor_names)
    counts = collapse_duplicate_symbols(counts)

    meta = pd.DataFrame(index=pd.Index(donor_names, name="donor"))
    meta["age"] = pd.Series(ages).groupby(donor_codes).first().values
    if sexes is not None:
        first_sex = pd.Series(sexes).groupby(donor_codes).first().str.lower()
        if first_sex.isin(["male", "female", "m", "f"]).any():
            meta["sex_male"] = first_sex.isin(["male", "m"]).astype(int).values
    meta["donor"] = meta.index

    fractions = None
    if ctypes is not None:
        tab = pd.crosstab(pd.Series(donor_codes, name="donor"), pd.Series(ctypes, name="cell_type"))
        tab.index = donor_names[tab.index]
        fractions = tab.div(tab.sum(axis=1), axis=0)
    return counts, meta, fractions


def log2_cpm(counts, library_size=None):
    lib = counts.sum(axis=0) if library_size is None else library_size
    return np.log2(counts.div(lib, axis=1) * 1e6 + 1)


def load_pseudobulk_cohort(name, cfg_key, cfg, panel, ens2sym, tissue, assay, lm22=None):
    section = cfg.get(cfg_key)
    if not section or not section.get("h5ad"):
        return None
    obs = section.get("obs", {})
    counts, meta, fractions = pseudobulk_h5ad(
        section["h5ad"], donor_col=obs.get("donor"), age_col=obs.get("age"),
        sex_col=obs.get("sex"), cell_type_col=obs.get("cell_type"),
        chunk_cells=int(section.get("chunk_cells", 20000)), ens2sym=ens2sym)
    meta = meta.dropna(subset=["age"])
    lib = counts.sum(axis=0)
    expr = log2_cpm(counts.loc[[g for g in panel if g in counts.index]], lib)
    linear = None
    if lm22 is not None:
        linear = counts.loc[counts.index.intersection(lm22.index)].div(lib, axis=1) * 1e6
    c = Cohort(name, expr, meta, linear=linear, panel=panel, tissue=tissue,
               assay=assay, age_label="exact (years)")
    c.cell_type_fractions = None if fractions is None else fractions.loc[c.meta.index]
    c.n_genes_total = counts.shape[0]
    return c


# ---------------------------------------------------------------------------
# 3.4  Cell-type deconvolution (NNLS against LM22)
# ---------------------------------------------------------------------------
def nnls_fractions(linear, lm22):
    """Non-negative least squares per sample on the genes shared with LM22;
    fractions normalised to sum to one."""
    shared = linear.index.intersection(lm22.index)
    S = lm22.loc[shared].values
    out = np.zeros((linear.shape[1], lm22.shape[1]))
    for j, col in enumerate(linear.columns):
        coef, _ = nnls(S, linear.loc[shared, col].values)
        out[j] = coef / coef.sum() if coef.sum() > 0 else coef
    return pd.DataFrame(out, index=linear.columns, columns=lm22.columns), len(shared)


# ---------------------------------------------------------------------------
# 3.5  Gene-wise age association (OLS: expression ~ age + sex)
# ---------------------------------------------------------------------------
def zscore(x):
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=1)
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def age_association(cohort, adjust_sex=True):
    """Standardised age coefficient (SD of expression per SD of age) per panel
    gene, sex-adjusted when the cohort has sex, BH-FDR across the panel."""
    age = zscore(cohort.age.values)
    design = pd.DataFrame({"age": age}, index=cohort.meta.index)
    if adjust_sex and cohort.has_sex:
        design["sex_male"] = cohort.meta["sex_male"].values
    X = sm.add_constant(design)
    rows = []
    for gene in cohort.genes:
        y = zscore(cohort.expr.loc[gene].values)
        fit = sm.OLS(y, X).fit()
        rows.append({"gene": gene, "group": GENE_GROUP[gene],
                     "beta_std": fit.params["age"], "p": fit.pvalues["age"],
                     "sex_adjusted": "sex_male" in design.columns})
    res = pd.DataFrame(rows).set_index("gene")
    res["fdr"] = multipletests(res["p"], method="fdr_bh")[1]
    return res.sort_values("fdr")


# ---------------------------------------------------------------------------
# 3.6  Sub-scores and the Polyamine-CIRBP Axis Score (PCAS)
# ---------------------------------------------------------------------------
def mean_z(expr, genes):
    present = [g for g in genes if g in expr.index]
    z = expr.loc[present].apply(zscore, axis=1, result_type="broadcast")
    return z.mean(axis=0)


def axis_scores(cohort):
    e = cohort.expr
    sub = pd.DataFrame({
        "PPAS": mean_z(e, POLYAMINE_BIOSYNTHESIS) - mean_z(e, POLYAMINE_CATABOLISM),
        "hypusination": mean_z(e, HYPUSINATION_SET),
        "CIRBP_module": mean_z(e, CIRBP_MODULE_SET),
        "SASP": mean_z(e, SASP_SET),
    })
    z = sub.apply(zscore, axis=0, result_type="broadcast")
    u, s, vt = np.linalg.svd(z.values - z.values.mean(axis=0), full_matrices=False)
    pc1, loadings = u[:, 0] * s[0], vt[0]
    if np.corrcoef(pc1, z["PPAS"])[0, 1] < 0:          # pre-specified orientation rule
        pc1, loadings = -pc1, -loadings
    scores = z.copy()
    scores["PCAS"] = pc1
    scores["PCAS_equal_weight"] = z.mean(axis=1)
    info = {"pc1_variance_explained": float(s[0] ** 2 / (s ** 2).sum()),
            "loadings": dict(zip(sub.columns, np.round(loadings, 3)))}
    return scores, info


def score_age_table(scores, age):
    rows = []
    for col in scores.columns:
        r, p = stats.pearsonr(scores[col], age)
        rows.append({"score": col, "pearson_r": r, "p": p, "r2": r ** 2})
    return pd.DataFrame(rows).set_index("score")


# ---------------------------------------------------------------------------
# 3.7  Machine-learning age clock (10-fold CV, per-fold reporting)
# ---------------------------------------------------------------------------
def elastic_net():
    # 40-point automatic alpha grid; the keyword changed name in scikit-learn 1.7
    grid = {"n_alphas": 40} if "n_alphas" in ElasticNetCV.__init__.__code__.co_varnames else {"alphas": 40}
    return ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9], cv=5, max_iter=20000,
                        random_state=SEED, **grid)


def random_forest(**params):
    p = dict(RF_BASE); p.update(params)
    return RandomForestRegressor(random_state=SEED, n_jobs=-1, **p)


def feature_matrix(cohort, cells=False, sex=False):
    X = cohort.expr.T.copy()
    if sex and cohort.has_sex:
        X["sex_male"] = cohort.meta["sex_male"].values
    if cells and cohort.cell_fractions is not None:
        X = X.join(cohort.cell_fractions.add_prefix("cell_"))
    return X


def folds_for(n):
    return list(KFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.arange(n)))


def metrics(y, yhat):
    return {"r2": r2_score(y, yhat), "mae": mean_absolute_error(y, yhat),
            "pearson_r": stats.pearsonr(y, yhat)[0]}


def cross_validate(model_factory, X, y, folds):
    """Standardise on the training donors of each fold, predict the held-out
    donors, report per-fold and pooled out-of-fold metrics."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    oof = np.zeros(len(y))
    per_fold = []
    for k, (tr, te) in enumerate(folds, start=1):
        scaler = StandardScaler().fit(X[tr])
        model = model_factory().fit(scaler.transform(X[tr]), y[tr])
        oof[te] = model.predict(scaler.transform(X[te]))
        per_fold.append({"fold": k, "n_train": len(tr), "n_test": len(te),
                         **metrics(y[te], oof[te])})
    per_fold = pd.DataFrame(per_fold).set_index("fold")
    return {"oof": oof, "per_fold": per_fold, "pooled": metrics(y, oof)}


def run_clocks(cohort, feature_sets):
    """feature_sets: {label: (model_factory, X)}; returns {label: cv result}."""
    y = cohort.age.values
    folds = folds_for(len(y))
    return {label: cross_validate(factory, X, y, folds) for label, (factory, X) in feature_sets.items()}


# ---------------------------------------------------------------------------
# 3.7a  Random Forest hyper-parameter exploration
# ---------------------------------------------------------------------------
def rf_sweep(X, y):
    folds = folds_for(len(y))
    rows = []
    for param, values in RF_SWEEP.items():
        for v in values:
            res = cross_validate(lambda: random_forest(**{param: v}), X, y, folds)
            rows.append({"parameter": param, "value": str(v), **res["pooled"]})
    sweep = pd.DataFrame(rows)
    ranges = sweep.groupby("parameter")["r2"].agg(lambda s: s.max() - s.min())
    grid_rows, best = [], None
    for n in RF_GRID["n_estimators"]:
        for d in RF_GRID["max_depth"]:
            for leaf in RF_GRID["min_samples_leaf"]:
                for mf in RF_GRID["max_features"]:
                    params = dict(n_estimators=n, max_depth=d, min_samples_leaf=leaf, max_features=mf)
                    res = cross_validate(lambda: random_forest(**params), X, y, folds)
                    grid_rows.append({**{k: str(v) for k, v in params.items()}, **res["pooled"]})
                    if best is None or res["pooled"]["r2"] > best[1]["pooled"]["r2"]:
                        best = (params, res)
    return sweep, ranges, pd.DataFrame(grid_rows).sort_values("r2", ascending=False), best


# ---------------------------------------------------------------------------
# 3.8  Model interpretation (RF impurity importance, Elastic Net coefficients)
# ---------------------------------------------------------------------------
def importances(X, y):
    Xs = StandardScaler().fit_transform(np.asarray(X, dtype=float))
    rf = random_forest().fit(Xs, y)
    en = elastic_net().fit(Xs, y)
    out = pd.DataFrame({"rf_importance": rf.feature_importances_,
                        "elastic_net_coef": en.coef_}, index=X.columns)
    out["group"] = [GENE_GROUP.get(f, "covariate") for f in out.index]
    return out.sort_values("rf_importance", ascending=False)


# ---------------------------------------------------------------------------
# 3.9  Machinery-inflammation coupling (Pearson r, age-partial r)
# ---------------------------------------------------------------------------
def residualise(x, age):
    X = sm.add_constant(np.asarray(age, dtype=float))
    return sm.OLS(np.asarray(x, dtype=float), X).fit().resid


def coupling(cohort):
    e, age = cohort.expr, cohort.age.values
    mach, infl = mean_z(e, MACHINERY_SET).values, mean_z(e, INFLAMMATION_SET).values
    r, p = stats.pearsonr(mach, infl)
    r_partial = stats.pearsonr(residualise(mach, age), residualise(infl, age))[0]
    return {"cohort": cohort.name, "n": len(age), "r": r, "p": p,
            "r_partial_age": r_partial,
            "r_machinery_age": stats.pearsonr(mach, age)[0],
            "r_inflammation_age": stats.pearsonr(infl, age)[0],
            "machinery": mach, "inflammation": infl}


# ---------------------------------------------------------------------------
# 4.9-4.13  Robustness, concordance, ROC, stepwise model improvement
# ---------------------------------------------------------------------------
def bootstrap_ci(x, y, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    x, y, n = np.asarray(x), np.asarray(y), len(x)
    rs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        rs[i] = np.corrcoef(x[idx], y[idx])[0, 1]
    return np.percentile(rs, [2.5, 97.5])


def stratified_performance(cohort, oof):
    y = cohort.age.values
    rows = []
    if cohort.has_sex:
        for label, val in (("female", 0), ("male", 1)):
            m = cohort.meta["sex_male"].values == val
            rows.append({"stratum": label, "n": int(m.sum()), **metrics(y[m], oof[m])})
    tertile = pd.qcut(y, 3, labels=["young tertile", "middle tertile", "old tertile"])
    for label in tertile.categories:
        m = np.asarray(tertile == label)
        rows.append({"stratum": f"age: {label}", "n": int(m.sum()),
                     "mae": mean_absolute_error(y[m], oof[m])})
    return pd.DataFrame(rows)


def cell_fraction_age(fractions, age, min_abundance=0.005):
    kept = fractions.loc[:, fractions.mean(axis=0) > min_abundance]
    rows = []
    for ct in kept.columns:
        r, p = stats.pearsonr(kept[ct].values, age.values)
        rows.append({"cell_type": ct, "mean_fraction": kept[ct].mean(), "r_age": r, "p": p})
    res = pd.DataFrame(rows).set_index("cell_type")
    res["fdr"] = multipletests(res["p"], method="fdr_bh")[1]
    return res.sort_values("r_age")


def concordance(assoc_a, assoc_b, names=("GTEx", "OneK1K")):
    a, b = names
    m = assoc_a.join(assoc_b, lsuffix=f"_{a}", rsuffix=f"_{b}", how="inner")
    m["group"] = m[f"group_{a}"]
    sig_a, sig_b = m[f"fdr_{a}"] < FDR_ALPHA, m[f"fdr_{b}"] < FDR_ALPHA
    m["significance"] = np.select(
        [sig_a & sig_b, sig_a, sig_b], ["both", f"{a} only", f"{b} only"], "neither")
    m["same_direction"] = np.sign(m[f"beta_std_{a}"]) == np.sign(m[f"beta_std_{b}"])
    both = m[m["significance"] == "both"]
    summary = m.groupby("group").apply(lambda g: pd.Series({
        "n_shared": len(g),
        "significant_both": int((g["significance"] == "both").sum()),
        f"significant_{a}_only": int((g["significance"] == f"{a} only").sum()),
        f"significant_{b}_only": int((g["significance"] == f"{b} only").sum()),
        "neither": int((g["significance"] == "neither").sum()),
        "direction_consistency": (g.loc[g["significance"] == "both", "same_direction"].mean()
                                  if (g["significance"] == "both").any() else np.nan)}))
    overall = {
        "n_shared": len(m), "replicated": len(both),
        "replicated_same_direction": int(both["same_direction"].sum()),
        "direction_consistency": both["same_direction"].mean() if len(both) else np.nan,
        "pearson_r": stats.pearsonr(m[f"beta_std_{a}"], m[f"beta_std_{b}"])[0],
        "pearson_p": stats.pearsonr(m[f"beta_std_{a}"], m[f"beta_std_{b}"])[1],
        "spearman_rho": stats.spearmanr(m[f"beta_std_{a}"], m[f"beta_std_{b}"])[0],
        "spearman_p": stats.spearmanr(m[f"beta_std_{a}"], m[f"beta_std_{b}"])[1]}
    return m, summary, overall


def group_effect_sizes(assocs):
    """Mean |beta_std| per functional group, one column per cohort (Table 15)."""
    cols = {}
    for name, a in assocs.items():
        cols[f"n_{name}"] = a.groupby("group").size()
        cols[f"mean_abs_beta_{name}"] = a.groupby("group")["beta_std"].apply(lambda s: s.abs().mean())
    return pd.DataFrame(cols)


def roc_analysis(y, predictions):
    """predictions: {model: oof}. AUC / AP for the three binarisations."""
    rows, curves = [], {}
    median = np.median(y)
    for split, bounds in ROC_SPLITS.items():
        if bounds is None:
            keep, label = np.ones(len(y), bool), (y >= median).astype(int)
        else:
            lo, hi = bounds
            keep, label = (y < lo) | (y >= hi), (y >= hi).astype(int)
        for model, oof in predictions.items():
            s, l = oof[keep], label[keep]
            rows.append({"split": split, "model": model, "n": int(keep.sum()),
                         "auc": roc_auc_score(l, s), "average_precision": average_precision_score(l, s)})
            curves[(split, model)] = (roc_curve(l, s), precision_recall_curve(l, s), l.mean())
    return pd.DataFrame(rows), curves


def xgboost_search(X, y):
    from xgboost import XGBRegressor
    space = dict(n_estimators=[100, 200, 400, 600], max_depth=[2, 3, 4, 6],
                 learning_rate=[0.01, 0.03, 0.05, 0.1], subsample=[0.6, 0.8, 1.0],
                 colsample_bytree=[0.5, 0.7, 1.0], reg_alpha=[0, 0.1, 1.0],
                 reg_lambda=[0.5, 1.0, 2.0])
    search = RandomizedSearchCV(XGBRegressor(random_state=SEED, n_jobs=-1, verbosity=0),
                                space, n_iter=60, cv=5, scoring="r2", random_state=SEED, n_jobs=1)
    return search.fit(X, y).best_params_


def stepwise_pipeline(cohort, concordant_genes):
    """Results 4.13: Elastic Net (full) -> Elastic Net (concordant genes) ->
    XGBoost (randomised search in the inner loop) -> Ridge-stacked ensemble.
    All stages share the same outer 10-fold split; the stacker is trained on
    inner out-of-fold predictions of the training donors only."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("  xgboost not installed: skipping stages 3-4", file=sys.stderr)
        return None
    X_full = feature_matrix(cohort, sex=True)
    conc = [g for g in concordant_genes if g in X_full.columns]
    X_conc = X_full[conc + (["sex_male"] if "sex_male" in X_full.columns else [])]
    y = cohort.age.values
    folds = folds_for(len(y))
    oof = {k: np.zeros(len(y)) for k in ("elastic_net_full", "elastic_net_concordant", "xgboost", "stacked")}
    inner = KFold(5, shuffle=True, random_state=SEED)
    for tr, te in folds:
        sc_full, sc_conc = StandardScaler().fit(X_full.values[tr]), StandardScaler().fit(X_conc.values[tr])
        A_tr, A_te = sc_full.transform(X_full.values[tr]), sc_full.transform(X_full.values[te])
        B_tr, B_te = sc_conc.transform(X_conc.values[tr]), sc_conc.transform(X_conc.values[te])
        best = xgboost_search(A_tr, y[tr])
        base = [("elastic_net_full", elastic_net, A_tr, A_te),
                ("elastic_net_concordant", elastic_net, B_tr, B_te),
                ("xgboost", lambda: XGBRegressor(random_state=SEED, n_jobs=-1, verbosity=0, **best), A_tr, A_te)]
        inner_oof = np.column_stack([cross_val_predict(f(), Xtr, y[tr], cv=inner) for _, f, Xtr, _ in base])
        test_pred = np.column_stack([f().fit(Xtr, y[tr]).predict(Xte) for _, f, Xtr, Xte in base])
        for j, (name, *_) in enumerate(base):
            oof[name][te] = test_pred[:, j]
        oof["stacked"][te] = Ridge(alpha=1.0).fit(inner_oof, y[tr]).predict(test_pred)
    summary = pd.DataFrame({k: metrics(y, v) for k, v in oof.items()}).T
    summary["n_features"] = [X_full.shape[1], X_conc.shape[1], X_full.shape[1], 3]
    return {"oof": oof, "summary": summary, "n_concordant_used": len(conc)}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
GROUP_COLORS = {
    "polyamine": "#1b9e77", "hypusination_machinery": "#d95f02",
    "hypusination_polyproline_targets": "#e6ab02", "cirbp_axis_core": "#000000",
    "cirbp_direct_targets": "#7570b3", "sasp_signature": "#e7298a",
    "nfkb_targets": "#66a61e", "senescence_markers": "#a6761d",
}


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_volcano(assoc, title, path):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sig = assoc[assoc["fdr"] < FDR_ALPHA]
    for grp, sub in assoc.groupby("group"):
        ax.scatter(sub["beta_std"], -np.log10(sub["p"]), s=28, alpha=0.85,
                   color=GROUP_COLORS.get(grp, "grey"), label=f"{grp} ({len(sub)})")
    if len(sig):
        ax.axhline(-np.log10(sig["p"].max()), ls="--", color="grey", lw=1)
    for gene, row in assoc.nsmallest(10, "fdr").iterrows():
        ax.annotate(gene, (row["beta_std"], -np.log10(row["p"])), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.axvline(0, color="lightgrey", lw=1)
    ax.set_xlabel("standardised age coefficient (β)")
    ax.set_ylabel("-log10 p")
    ax.set_title(f"{title}: {len(sig)}/{len(assoc)} genes at FDR<{FDR_ALPHA}")
    ax.legend(fontsize=7, loc="best")
    save(fig, path)


def fig_pcas_by_decade(scores, meta, path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bins = meta["age_bin"] if "age_bin" in meta.columns else meta["age"].astype(int).astype(str)
    order = sorted(bins.unique())
    ax.boxplot([scores.loc[bins == b, "PCAS"].values for b in order])
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order)
    ax.set_xlabel("GTEx age decade")
    ax.set_ylabel("PCAS")
    save(fig, path)


def fig_per_fold(results, path, metric_names=("r2", "mae", "pearson_r")):
    fig, axes = plt.subplots(1, len(metric_names), figsize=(4.5 * len(metric_names), 4))
    width = 0.8 / len(results)
    for ax, met in zip(axes, metric_names):
        for i, (label, res) in enumerate(results.items()):
            pf = res["per_fold"]
            ax.bar(pf.index + i * width - 0.4 + width / 2, pf[met], width, label=label)
        ax.set_xlabel("fold"); ax.set_ylabel(met); ax.set_xticks(range(1, N_FOLDS + 1))
        if met == "r2":
            ax.axhline(0, color="grey", lw=0.8)
    axes[0].legend(fontsize=8)
    save(fig, path)


def fig_fold_r2_lines(results, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, res in results.items():
        ax.plot(res["per_fold"].index, res["per_fold"]["r2"], marker="o",
                label=f"{label} (pooled R²={res['pooled']['r2']:.3f})")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("fold"); ax.set_ylabel("held-out R²"); ax.set_xticks(range(1, N_FOLDS + 1))
    ax.legend(fontsize=8)
    save(fig, path)


def fig_rf_ranges(ranges, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    r = ranges.sort_values()
    ax.barh(r.index, r.values, color="#7570b3")
    ax.set_xlabel("range of pooled out-of-fold R² across explored values")
    save(fig, path)


def fig_clock_scatter(ax, y, oof, title):
    ax.scatter(y, oof, s=12, alpha=0.5)
    lims = [min(y.min(), oof.min()), max(y.max(), oof.max())]
    ax.plot(lims, lims, "--", color="grey", lw=1)
    m = metrics(y, oof)
    ax.set_xlabel("chronological age"); ax.set_ylabel("predicted age (out-of-fold)")
    ax.set_title(f"{title}\nR²={m['r2']:.3f}, MAE={m['mae']:.2f}, r={m['pearson_r']:.3f}", fontsize=9)


def fig_clock_and_models(y, clocks, path, primary):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig_clock_scatter(axes[0], y, clocks[primary]["oof"], primary)
    labels = list(clocks)
    axes[1].bar(labels, [clocks[k]["pooled"]["r2"] for k in labels], color="#1b9e77")
    axes[1].set_ylabel("pooled out-of-fold R²")
    axes[1].tick_params(axis="x", labelrotation=20, labelsize=8)
    save(fig, path)


def fig_cohort_summary(cohort, scores, clock, path):
    y = cohort.age.values
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].hist(y, bins=20, color="#7570b3"); axes[0].set_xlabel("age"); axes[0].set_ylabel("donors")
    r, p = stats.pearsonr(scores["PCAS"], y)
    axes[1].scatter(y, scores["PCAS"], s=12, alpha=0.5); axes[1].set_xlabel("age"); axes[1].set_ylabel("PCAS")
    axes[1].set_title(f"r={r:+.3f}, p={p:.2g}", fontsize=9)
    fig_clock_scatter(axes[2], y, clock["oof"], "Elastic Net clock")
    fig.suptitle(f"{cohort.name} (n={len(y)})")
    save(fig, path)


def fig_importances(imp, path, top=20):
    top_imp = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.barh(top_imp.index, top_imp["rf_importance"],
            color=[GROUP_COLORS.get(g, "grey") for g in top_imp["group"]])
    ax.set_xlabel("Random Forest impurity importance")
    save(fig, path)


def fig_coupling(couplings, path):
    fig, axes = plt.subplots(1, len(couplings), figsize=(5 * len(couplings), 4.2), squeeze=False)
    for ax, c in zip(axes[0], couplings):
        ax.scatter(c["machinery"], c["inflammation"], s=12, alpha=0.5)
        ax.set_xlabel("axis machinery score"); ax.set_ylabel("inflammation score")
        ax.set_title(f"{c['cohort']} (n={c['n']}): r={c['r']:+.3f}, p={c['p']:.1e}", fontsize=9)
    save(fig, path)


def fig_cell_fractions(res, path):
    fig, ax = plt.subplots(figsize=(8, max(4, 0.32 * len(res))))
    colors = ["#d95f02" if r > 0 else "#1b9e77" for r in res["r_age"]]
    ax.barh(res.index, res["r_age"], color=colors)
    for i, (ct, row) in enumerate(res.iterrows()):
        if row["fdr"] < FDR_ALPHA:
            ax.text(row["r_age"], i, " *", va="center", fontsize=8)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Pearson r with age (* FDR<0.05)")
    ax.tick_params(axis="y", labelsize=7)
    save(fig, path)


def fig_concordance(merged, summary, overall, names, path):
    a, b = names
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    palette = {"both": "#d95f02", f"{a} only": "#7570b3", f"{b} only": "#1b9e77", "neither": "lightgrey"}
    for cat, sub in merged.groupby("significance"):
        axes[0].scatter(sub[f"beta_std_{a}"], sub[f"beta_std_{b}"], s=22, color=palette[cat], label=cat)
    axes[0].axhline(0, color="grey", lw=0.8); axes[0].axvline(0, color="grey", lw=0.8)
    axes[0].set_xlabel(f"β ({a})"); axes[0].set_ylabel(f"β ({b})")
    axes[0].set_title(f"r={overall['pearson_r']:+.3f}, p={overall['pearson_p']:.2g}", fontsize=9)
    axes[0].legend(fontsize=8)
    s = summary["direction_consistency"].dropna()
    axes[1].bar(s.index, s.values, color=[GROUP_COLORS.get(g, "grey") for g in s.index])
    axes[1].axhline(overall["direction_consistency"], ls="--", color="grey")
    axes[1].set_ylabel("direction consistency (replicated genes)")
    axes[1].tick_params(axis="x", labelrotation=35, labelsize=7)
    save(fig, path)


def fig_roc(curves, table, path):
    splits = list(ROC_SPLITS)
    fig, axes = plt.subplots(2, len(splits), figsize=(5 * len(splits), 9))
    for j, split in enumerate(splits):
        for (s, model), (roc, pr, prevalence) in curves.items():
            if s != split:
                continue
            row = table[(table["split"] == split) & (table["model"] == model)].iloc[0]
            axes[0, j].plot(roc[0], roc[1], label=f"{model} (AUC={row['auc']:.3f})")
            axes[1, j].plot(pr[1], pr[0], label=f"{model} (AP={row['average_precision']:.3f})")
        axes[0, j].plot([0, 1], [0, 1], "--", color="grey"); axes[1, j].axhline(prevalence, ls="--", color="grey")
        axes[0, j].set_title(split, fontsize=9); axes[0, j].set_xlabel("FPR"); axes[0, j].set_ylabel("TPR")
        axes[1, j].set_xlabel("recall"); axes[1, j].set_ylabel("precision")
        axes[0, j].legend(fontsize=7); axes[1, j].legend(fontsize=7)
    save(fig, path)


def fig_stepwise(step, y, path):
    s = step["summary"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].bar(s.index, s["r2"], color="#7570b3"); axes[0].axhline(s.loc["elastic_net_full", "r2"], ls=":", color="grey")
    axes[0].set_ylabel("pooled out-of-fold R²")
    axes[1].bar(s.index, s["mae"], color="#1b9e77"); axes[1].set_ylabel("MAE (years)")
    for ax in axes[:2]:
        ax.tick_params(axis="x", labelrotation=20, labelsize=8)
    axes[2].scatter(y, step["oof"]["elastic_net_full"], s=10, alpha=0.4, label="Elastic Net")
    axes[2].scatter(y, step["oof"]["stacked"], s=10, alpha=0.4, color="red", label="stacked")
    lims = [y.min(), y.max()]; axes[2].plot(lims, lims, "--", color="grey"); axes[2].legend(fontsize=8)
    axes[2].set_xlabel("chronological age"); axes[2].set_ylabel("predicted age")
    save(fig, path)


def fig_bias(cohorts, clock_summary, assocs, path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, c in zip(axes[0], cohorts):
        ax.hist(c.age.values, bins=20, color="#7570b3"); ax.set_title(f"{c.name} (n={len(c.age)})"); ax.set_xlabel("age")
    for c in cohorts:
        row = clock_summary[(clock_summary["cohort"] == c.name) & (clock_summary["model"] == "Elastic Net")
                            & clock_summary["features"].isin(["panel", "panel+sex"])]
        if len(row):
            axes[1, 0].scatter(c.age.std(), row["r2"].iloc[0], s=60)
            axes[1, 0].annotate(c.name, (c.age.std(), row["r2"].iloc[0]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[1, 0].set_xlabel("SD of cohort age (years)"); axes[1, 0].set_ylabel("Elastic Net pooled R²")
    if "GTEx" in assocs and "OneK1K" in assocs:
        nf = [g for g in PANEL_GROUPS["nfkb_targets"] if g in assocs["GTEx"].index and g in assocs["OneK1K"].index]
        x = np.arange(len(nf))
        axes[1, 1].bar(x - 0.2, assocs["GTEx"].loc[nf, "beta_std"], 0.4, label="GTEx (post mortem)")
        axes[1, 1].bar(x + 0.2, assocs["OneK1K"].loc[nf, "beta_std"], 0.4, label="OneK1K (living donors)")
        axes[1, 1].set_xticks(x); axes[1, 1].set_xticklabels(nf, rotation=60, fontsize=7)
        axes[1, 1].axhline(0, color="grey", lw=0.8); axes[1, 1].set_ylabel("β (age)"); axes[1, 1].legend(fontsize=8)
    for c in cohorts:
        axes[1, 2].hist(c.age.values, bins=20, density=True, alpha=0.4, label=c.name)
    axes[1, 2].set_xlabel("age"); axes[1, 2].set_ylabel("density"); axes[1, 2].legend(fontsize=8)
    save(fig, path)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def write(df, out, name, **kw):
    df.to_csv(os.path.join(out, "tables", name), **kw)


def clock_summary_row(cohort, features, model, res):
    return {"cohort": cohort.name, "features": features, "model": model, "n": len(cohort.age),
            **res["pooled"], "fold_r2_mean": res["per_fold"]["r2"].mean(),
            "fold_r2_sd": res["per_fold"]["r2"].std()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="JSON file with input paths (see README.md)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--skip-rf-sweep", action="store_true")
    ap.add_argument("--skip-stepwise", action="store_true")
    args = ap.parse_args(argv)
    with open(args.config) as fh:
        cfg = json.load(fh)
    for sub in ("tables", "figures"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    out, fig_dir = args.out, os.path.join(args.out, "figures")
    log = {"seed": SEED, "n_folds": N_FOLDS, "panel_106": len(PANEL_106), "panel_onek1k": len(PANEL_ONEK1K)}

    # ---- 3.2 / 3.4 inputs -------------------------------------------------
    lm22 = read_lm22(cfg["lm22"]) if cfg.get("lm22") else None
    gtex = load_gtex(cfg) if cfg.get("gtex_gct") else None
    ens2sym = gtex.ens2sym if gtex else None
    if gtex is not None and lm22 is not None:
        gtex.cell_fractions, log["gtex_lm22_shared_genes"] = nnls_fractions(gtex.linear, lm22)
        gtex.cell_fractions.to_csv(os.path.join(out, "tables", "gtex_nnls_cell_fractions.csv"))
    gse_full = load_pseudobulk_cohort("GSE233321_full", "gse233321", cfg, PANEL_106, ens2sym,
                                      "PBMC (pseudobulk scRNA)", "scRNA-seq -> pseudobulk", lm22)
    gse = gse_full.subset(gse_full.age >= 18, "GSE233321") if gse_full is not None else None
    onek1k = load_pseudobulk_cohort("OneK1K", "onek1k", cfg, PANEL_ONEK1K, ens2sym,
                                    "Blood (pseudobulk scRNA)", "scRNA-seq -> pseudobulk")
    primary = [c for c in (gtex, gse, onek1k) if c is not None]
    if not primary:
        sys.exit("no cohort inputs found in config")
    print("cohorts:", ", ".join(f"{c.name} (n={len(c.age)}, {len(c.genes)} panel genes)" for c in primary))

    # ---- Table 1 ----------------------------------------------------------
    write(pd.DataFrame([{
        "cohort": c.name, "donors": len(c.age), "tissue": c.tissue, "assay": c.assay,
        "age_encoding": c.age_label, "age_min": c.age.min(), "age_max": c.age.max(),
        "age_mean": round(c.age.mean(), 1), "age_sd": round(c.age.std(), 1),
        "female": int((c.meta["sex_male"] == 0).sum()) if c.has_sex else "n/a",
        "male": int((c.meta["sex_male"] == 1).sum()) if c.has_sex else "n/a",
        "panel_genes_detected": len(c.genes), "panel_version": len(c.panel),
        "undetected": ";".join(c.undetected)}
        for c in primary]), out, "table01_cohorts.csv", index=False)

    # ---- 3.5 / 4.2  gene-wise age association (Tables 2, 17-19; Figures 4-6)
    assocs = {}
    for c, fig_no, tab_no in ((gtex, 4, 17), (gse, 5, 19), (onek1k, 6, 18)):
        if c is None:
            continue
        assocs[c.name] = age_association(c)
        write(assocs[c.name], out, f"table{tab_no}_{c.name}_age_association.csv")
        fig_volcano(assocs[c.name], c.name, os.path.join(fig_dir, f"fig{fig_no:02d}_volcano_{c.name}.png"))
        log[f"{c.name}_significant_fdr05"] = int((assocs[c.name]["fdr"] < FDR_ALPHA).sum())
    if gse_full is not None:
        assocs["GSE233321_full"] = age_association(gse_full)
        write(assocs["GSE233321_full"], out, "table19b_GSE233321_full_age_association.csv")
        log["GSE233321_full_significant_fdr05"] = int((assocs["GSE233321_full"]["fdr"] < FDR_ALPHA).sum())
    if gtex is not None:
        write(assocs["GTEx"].head(20), out, "table02_GTEx_top20_age_genes.csv")
        log["GTEx_median_abs_beta"] = float(assocs["GTEx"]["beta_std"].abs().median())

    # ---- 3.6 / 4.3  sub-scores and PCAS (Table 3, Figure 7) ---------------
    scores = {}
    for c in primary + ([gse_full] if gse_full is not None else []):
        scores[c.name], info = axis_scores(c)
        log[f"{c.name}_pcas"] = info
        scores[c.name].to_csv(os.path.join(out, "tables", f"scores_{c.name}.csv"))
        write(score_age_table(scores[c.name], c.age.values), out, f"table03_{c.name}_score_age.csv")
    if gtex is not None:
        fig_pcas_by_decade(scores["GTEx"], gtex.meta, os.path.join(fig_dir, "fig07_GTEx_PCAS_by_decade.png"))

    # ---- 3.7 / 4.4-4.6  age clocks (Tables 4, 5, 8; Figures 8, 9, 11-13) --
    clocks, rows = {}, []
    if gtex is not None:
        sets = {"Elastic Net | panel": (elastic_net, feature_matrix(gtex)),
                "Random Forest | panel": (random_forest, feature_matrix(gtex))}
        if gtex.cell_fractions is not None:
            sets["Elastic Net | panel+cells+sex"] = (elastic_net, feature_matrix(gtex, cells=True, sex=True))
        clocks["GTEx"] = run_clocks(gtex, sets)
        fig_clock_and_models(gtex.age.values, clocks["GTEx"],
                             os.path.join(fig_dir, "fig11_GTEx_clock_and_models.png"), "Elastic Net | panel")
    if gse is not None:
        clocks["GSE233321"] = run_clocks(gse, {"Elastic Net | panel": (elastic_net, feature_matrix(gse)),
                                               "Random Forest | panel": (random_forest, feature_matrix(gse))})
        write(clocks["GSE233321"]["Elastic Net | panel"]["per_fold"], out, "table04_GSE233321_elastic_net_folds.csv")
        write(clocks["GSE233321"]["Random Forest | panel"]["per_fold"], out, "table05_GSE233321_random_forest_folds.csv")
        if not args.skip_rf_sweep:                                     # 3.7a / 4.4b
            sweep, ranges, grid, (best_params, best_res) = rf_sweep(feature_matrix(gse), gse.age.values)
            write(sweep, out, "table06_GSE233321_rf_sweep.csv", index=False)
            write(grid, out, "table06b_GSE233321_rf_grid.csv", index=False)
            write(best_res["per_fold"], out, "table07_GSE233321_rf_tuned_folds.csv")
            clocks["GSE233321"]["Random Forest (tuned) | panel"] = best_res
            log["GSE233321_rf_best_params"] = {k: str(v) for k, v in best_params.items()}
            fig_rf_ranges(ranges, os.path.join(fig_dir, "fig10_rf_hyperparameter_ranges.png"))
        fig_per_fold(clocks["GSE233321"], os.path.join(fig_dir, "fig08_GSE233321_per_fold_metrics.png"))
        fig_fold_r2_lines(clocks["GSE233321"], os.path.join(fig_dir, "fig09_GSE233321_fold_r2.png"))
        fig_cohort_summary(gse, scores["GSE233321"], clocks["GSE233321"]["Elastic Net | panel"],
                           os.path.join(fig_dir, "fig12_GSE233321_summary.png"))
    if onek1k is not None:
        clocks["OneK1K"] = run_clocks(onek1k, {
            "Elastic Net | panel+sex": (elastic_net, feature_matrix(onek1k, sex=True)),
            "Random Forest | panel+sex": (random_forest, feature_matrix(onek1k, sex=True))})
        fig_cohort_summary(onek1k, scores["OneK1K"], clocks["OneK1K"]["Elastic Net | panel+sex"],
                           os.path.join(fig_dir, "fig13_OneK1K_summary.png"))
    for cname, res_by_label in clocks.items():
        c = {x.name: x for x in primary}[cname]
        for label, res in res_by_label.items():
            model, features = [s.strip() for s in label.split("|")]
            rows.append(clock_summary_row(c, features, model, res))
            pd.DataFrame({"age": c.age.values, "predicted": res["oof"]}, index=c.meta.index).to_csv(
                os.path.join(out, "tables", f"oof_{cname}_{model.replace(' ', '_')}_{features}.csv"))
    clock_summary = pd.DataFrame(rows)
    write(clock_summary, out, "table08_clock_summary.csv", index=False)

    # ---- 3.8 / 4.7  model interpretation (Tables 9, 20; Figure 14) --------
    if gtex is not None:
        imp = importances(feature_matrix(gtex), gtex.age.values)
        write(imp, out, "table20_GTEx_importances_all.csv")
        write(imp.head(15), out, "table09_GTEx_importances_top15.csv")
        fig_importances(imp, os.path.join(fig_dir, "fig14_GTEx_importances_top20.png"))

    # ---- 3.9 / 4.8  machinery-inflammation coupling (Table 10, Figure 15) -
    couplings = [coupling(c) for c in primary + ([gse_full] if gse_full is not None else [])]
    write(pd.DataFrame([{k: v for k, v in c.items() if k not in ("machinery", "inflammation")}
                        for c in couplings]), out, "table10_coupling.csv", index=False)
    fig_coupling([c for c in couplings if c["cohort"].startswith("GSE233321")] or couplings,
                 os.path.join(fig_dir, "fig15_coupling.png"))

    # ---- 4.9  immune composition vs age (OneK1K; Table 11, Figure 16) -----
    if onek1k is not None and getattr(onek1k, "cell_type_fractions", None) is not None:
        cf = cell_fraction_age(onek1k.cell_type_fractions, onek1k.age)
        write(cf, out, "table11_OneK1K_cell_fractions_age.csv")
        fig_cell_fractions(cf, os.path.join(fig_dir, "fig16_OneK1K_cell_fractions.png"))

    # ---- 4.10  bootstrap CIs and stratified performance (Tables 12, 13) ---
    boot, strat = [], []
    for c in primary:
        cp = [x for x in couplings if x["cohort"] == c.name][0]
        for label, (x, y) in (("PCAS ~ age", (scores[c.name]["PCAS"].values, c.age.values)),
                              ("machinery ~ inflammation", (cp["machinery"], cp["inflammation"]))):
            lo, hi = bootstrap_ci(x, y)
            boot.append({"correlation": label, "cohort": c.name, "r": np.corrcoef(x, y)[0, 1],
                         "ci95_low": lo, "ci95_high": hi})
        en = [k for k in clocks.get(c.name, {}) if k.startswith("Elastic Net")]
        if en:
            s = stratified_performance(c, clocks[c.name][en[0]]["oof"])
            s.insert(0, "cohort", c.name)
            strat.append(s)
    write(pd.DataFrame(boot), out, "table12_bootstrap_ci.csv", index=False)
    if strat:
        write(pd.concat(strat), out, "table13_stratified_performance.csv", index=False)

    # ---- 4.11  cross-cohort concordance (Tables 14, 15; Figure 17) --------
    concordant_genes = []
    if gtex is not None and onek1k is not None:
        merged, summary, overall = concordance(assocs["GTEx"], assocs["OneK1K"])
        write(merged, out, "table14b_concordance_genes.csv")
        write(summary, out, "table14_concordance_groups.csv")
        log["concordance"] = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in overall.items()}
        concordant_genes = list(merged.index[merged["significance"] == "both"])
        fig_concordance(merged, summary, overall, ("GTEx", "OneK1K"), os.path.join(fig_dir, "fig17_concordance.png"))
    write(group_effect_sizes({k: v for k, v in assocs.items() if k in ("GTEx", "OneK1K", "GSE233321")}),
          out, "table15_group_effect_sizes.csv")

    # ---- 4.12 / 4.13  stepwise improvement and ROC (Figures 18, 19) -------
    if onek1k is not None and concordant_genes and not args.skip_stepwise:
        step = stepwise_pipeline(onek1k, concordant_genes)
        if step is not None:
            write(step["summary"], out, "table_stepwise_OneK1K.csv")
            pd.DataFrame({"age": onek1k.age.values, **step["oof"]}, index=onek1k.meta.index).to_csv(
                os.path.join(out, "tables", "oof_OneK1K_stepwise.csv"))
            log["stepwise_n_concordant_used"] = step["n_concordant_used"]
            roc_table, curves = roc_analysis(onek1k.age.values, step["oof"])
            write(roc_table, out, "table_roc_OneK1K.csv", index=False)
            fig_roc(curves, roc_table, os.path.join(fig_dir, "fig18_OneK1K_roc_pr.png"))
            fig_stepwise(step, onek1k.age.values, os.path.join(fig_dir, "fig19_OneK1K_stepwise.png"))

    # ---- Annex F  dataset bias (Figure 21, panels A-E) --------------------
    fig_bias(primary, clock_summary, assocs, os.path.join(fig_dir, "fig21_dataset_bias.png"))

    with open(os.path.join(out, "run_summary.json"), "w") as fh:
        json.dump(log, fh, indent=2, default=str)
    print(f"done: tables and figures written to {out}/")


if __name__ == "__main__":
    main()
