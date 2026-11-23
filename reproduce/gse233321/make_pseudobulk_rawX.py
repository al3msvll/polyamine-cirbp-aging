"""Donor-level pseudobulk for GSE233321 (thesis Section 3.4a).

Sums the integer UMI counts stored in raw/X of GEO GSE233321_all_PBMCs.h5ad
(1,191,563 cells x 21,222 genes) per donor (obs 'sample_id'), reading 100,000 cells
at a time (peak memory ~4 GB). Donors are sorted alphabetically. Note that X in this
file holds 2,000 scaled highly variable genes and is not used.

Usage:
    python make_pseudobulk_rawX.py <GSE233321_all_PBMCs.h5ad> [out_dir]
Writes GSE233321_pseudobulk_counts_rawX.parquet (genes x donors) and
GSE233321_donor_metadata.csv (donor, age, n_cells).
"""
import os
import sys

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pseudobulk import _obs_column, _decode

CHUNK = 100_000


def main(h5ad, out_dir):
    with h5py.File(h5ad, "r") as f:
        n_cells, n_genes = (int(v) for v in f["raw/X"].attrs["shape"])
        indptr = f["raw/X/indptr"][:]
        indices, data = f["raw/X/indices"], f["raw/X/data"]
        symbols = _decode(f["raw/var/gene_symbol"][:])
        donors = _obs_column(f, "sample_id")
        ages = pd.to_numeric(pd.Series(_obs_column(f, "Age_in_yrs")), errors="coerce").values

        uniq, inv = np.unique(donors, return_inverse=True)
        sums = np.zeros((n_genes, len(uniq)))
        for c0 in range(0, n_cells, CHUNK):
            c1 = min(c0 + CHUNK, n_cells)
            lo, hi = int(indptr[c0]), int(indptr[c1])
            block = sparse.csr_matrix((data[lo:hi].astype(np.float64), indices[lo:hi], indptr[c0:c1 + 1] - lo),
                                      shape=(c1 - c0, n_genes))
            donor_matrix = sparse.csr_matrix((np.ones(c1 - c0), (inv[c0:c1], np.arange(c1 - c0))),
                                             shape=(len(uniq), c1 - c0))
            sums += (donor_matrix @ block).T.toarray()
            print(f"cells {c1:,}/{n_cells:,}", flush=True)

    counts = pd.DataFrame(sums, index=symbols, columns=uniq).groupby(level=0).sum()
    counts.to_parquet(os.path.join(out_dir, "GSE233321_pseudobulk_counts_rawX.parquet"))
    meta = (pd.DataFrame({"donor": donors, "age": ages}).groupby("donor")
            .agg(age=("age", "median"), n_cells=("age", "size")))
    meta.to_csv(os.path.join(out_dir, "GSE233321_donor_metadata.csv"))
    print(f"{counts.shape[0]} genes x {counts.shape[1]} donors written to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ".")
