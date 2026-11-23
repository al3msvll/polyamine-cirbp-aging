"""
pseudobulk.py — donor-level aggregation of the single-cell cohorts.

GSE233321 (GEO, ~1.19M PBMCs, 167 donors) and OneK1K (CELLxGENE,
1,248,980 cells, 981 donors): sum raw counts per donor, normalise to CPM
(donor library size), log2 transform (thesis Section 3.4a).

Memory strategy: the h5ad is read with h5py (no anndata object); cells are
processed in blocks of 100k. For each block the CSR slice is assembled as a
scipy sparse matrix and multiplied by the block's donor indicator matrix
(X.T @ D), so peak RAM stays around 3-4 GB regardless of cohort size.
"""

import numpy as np
import pandas as pd
import h5py
from scipy import sparse

CHUNK = 100_000  # cells per block


def _decode(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


def _obs_column(f, key):
    """Return obs[key] resolved through categorical codes, values as strings
    when categorical, else raw values. Handles both anndata layouts:
    obs/<key> as dataset (with optional obs/__categories/<key>) and
    obs/<key>/codes + obs/<key>/categories."""
    node = f[f"obs/{key}"]
    if isinstance(node, h5py.Group):
        # categorical stored in-place: codes + categories
        cats = _decode(node["categories"][:])
        codes = node["codes"][:]
        return np.array([cats[i] for i in codes])
    raw = node[:]
    if "__categories" in f["obs"] and key in f["obs/__categories"]:
        cats = _decode(f[f"obs/__categories/{key}"][:])
        return np.array([cats[i] for i in raw])
    if raw.dtype.kind in "SU":
        return _decode(raw)
    return raw


def pseudobulk_h5ad(path, donor_key="donor_id", chunk=CHUNK, obs_keys=()):
    """
    Aggregate raw counts per donor.

    obs_keys: extra obs columns to resolve per cell (read in the same pass).
    Returns (counts genes x donors, meta DataFrame indexed by donor with
    n_cells and total UMI, obs_dict mapping key -> per-cell values).
    """
    with h5py.File(path, "r") as f:
        enc = dict(f["X"].attrs)
        n_cells, n_genes = (int(v) for v in np.atleast_1d(enc["shape"]))
        indptr_cache = f["X/indptr"][:]      # small: n_cells+1 ints
        indices_ds = f["X/indices"]          # lazy: sliced per block
        data_ds = f["X/data"]                # lazy: sliced per block
        genes = _decode(f["var/_index"][:])

        donors = _obs_column(f, donor_key)
        obs_resolved = {k: _obs_column(f, k) for k in obs_keys}

        uniq, inv = np.unique(donors, return_inverse=True)
        n_d = len(uniq)

        sums = np.zeros((n_genes, n_d), dtype=np.float64)
        n_cells_d = np.zeros(n_d, dtype=np.int64)
        umi_d = np.zeros(n_d, dtype=np.float64)

        for c0 in range(0, n_cells, chunk):
            c1 = min(c0 + chunk, n_cells)
            lo, hi = int(indptr_cache[c0]), int(indptr_cache[c1])
            block = sparse.csc_matrix(
                (data_ds[lo:hi], indices_ds[lo:hi],
                 indptr_cache[c0:c1 + 1] - indptr_cache[c0]),
                shape=(n_genes, c1 - c0),
            )
            inv_b = inv[c0:c1]
            D = sparse.csc_matrix(
                (np.ones(c1 - c0), inv_b,
                 np.arange(c1 - c0 + 1, dtype=np.int64)),
                shape=(n_d, c1 - c0),
            )
            sums += (block @ D.T).toarray()
            np.add.at(n_cells_d, inv_b, 1)
            np.add.at(umi_d, inv_b, np.asarray(block.sum(axis=0)).ravel())
            print(f"  cells {c1}/{n_cells}", flush=True)

    counts = pd.DataFrame(sums, index=genes, columns=uniq)
    meta = pd.DataFrame({"n_cells": n_cells_d, "total_UMI": umi_d}, index=uniq)
    return counts, meta, obs_resolved


def log2cpm(counts):
    """CPM normalisation by donor library size, then log2 (thesis 3.4a)."""
    lib = counts.sum(axis=0).replace(0, np.nan)
    cpm = counts / lib * 1e6
    return np.log2(cpm + 1)


if __name__ == "__main__":
    import sys
    print("smoke: module imports OK; called by runner scripts")
