"""Gene sets and sub-score definitions behind thesis Table 3 (GTEx whole blood).

PPAS          mean z(first five 'polyamine' genes) - mean z(remaining two: SMOX, PAOX)
Hypusination  mean z(EIF5A, EIF5A2, DHPS, DOHH)
CIRBP module  mean z(CIRBP + first ten 'cirbp_direct_targets')
SASP          mean z(first eight 'sasp_signature' genes)
PCAS          first principal component of the four standardised sub-scores,
              oriented to correlate positively with age
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PANEL = {
    "polyamine": [
        "ODC1", "SRM", "SMS", "AMD1", "SAT1", "SMOX", "PAOX",
    ],
    "hypusination_machinery": [
        "EIF5A", "EIF5A2", "DHPS", "DOHH",
    ],
    "hypusination_polyproline_targets": [
        # Polyproline-motif-containing translation targets of hypusinated eIF5A
        "ATXN1", "ATXN2", "ATXN3", "ATXN7", "FMR1", "MEF2A", "MEF2C", "CALR",
        "TIMP1", "PSEN1", "PCNA", "BCL2L11", "CDK1", "CDK6", "DRAM2",
        "EIF4G2", "FOXM1", "MAP2K3", "PSMD7", "RAB7A",
    ],
    "cirbp_axis_core": ["CIRBP"],
    "cirbp_direct_targets": [
        # Direct CIRBP-stabilised mRNAs (Liao 2017, Lujan 2018, Indacochea 2021)
        "TNF", "IL6", "VEGFA", "HIF1A", "HMGB1", "CCND1", "BCL2", "BAX",
        "GADD45B", "MYC", "CCNB1", "CST3", "MMP9", "NFE2L2", "XIAP", "BIRC5",
        "E2F1", "RPS3", "HSPA1A", "HSPB1", "ATF3", "DUSP1", "FOS", "JUN",
        "EGR1", "NR4A1", "CDKN1A", "MDM2",
    ],
    "sasp_signature": [
        # Core SASP from Basisty et al. 2020
        "IL6", "CXCL8", "IL1A", "IL1B", "TNF", "CCL2", "CCL20", "CXCL1",
        "CXCL2", "CXCL10", "MMP1", "MMP3", "MMP10", "MMP12", "TIMP1", "IGFBP3",
        "IGFBP7", "VEGFA", "HGF", "PLAUR", "SERPINE1", "SERPINB2", "ICAM1",
        "FAS", "FASLG", "ANGPTL4", "PTGS2", "NPPB", "NRG1", "LTA",
    ],
    "nfkb_targets": [
        # Gilmore lab NF-κB target gene curation
        "NFKB1", "NFKB2", "RELA", "RELB", "REL", "NFKBIA", "NFKBIB", "TLR2",
        "TLR4", "MYD88", "IRAK1", "IRAK4", "TRAF6", "TNFAIP3", "BIRC3",
    ],
    "senescence_markers": [
        "CDKN1A", "CDKN2A", "TP53", "MKI67", "LMNB1", "RB1",
    ],
}


def _mean_z(X, genes):
    g = [x for x in genes if x in X.columns]
    Z = (X[g] - X[g].mean()) / X[g].std()
    return Z.mean(axis=1)


def axis_scores(X, age):
    """X: samples x genes (log1p TPM); age: Series aligned to X. Returns (scores, pca)."""
    sub = pd.DataFrame({
        "PPAS": _mean_z(X, PANEL["polyamine"][:5]) - _mean_z(X, PANEL["polyamine"][5:]),
        "Hypusination": _mean_z(X, PANEL["hypusination_machinery"]),
        "CIRBP_module": _mean_z(X, ["CIRBP"] + PANEL["cirbp_direct_targets"][:10]),
        "SASP": _mean_z(X, PANEL["sasp_signature"][:8]),
    })
    pca = PCA(n_components=1)
    pc = pca.fit_transform(StandardScaler().fit_transform(sub))[:, 0]
    if np.corrcoef(pc, age)[0, 1] < 0:
        pc, pca.components_ = -pc, -pca.components_
    sub["PCAS"] = pc
    return sub, pca
