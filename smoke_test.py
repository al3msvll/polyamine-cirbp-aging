#!/usr/bin/env python3
"""Run the pipeline end to end on small synthetic inputs written in the
manuscript's file formats (GTEx .gct.gz + phenotypes, LM22.txt, two h5ad
files). Checks that every stage executes; the numbers it produces mean nothing.

    python smoke_test.py [--full] [--keep]

--full  also runs the RF sweep and the stepwise/XGBoost stage (slower)
--keep  print the temp folder and leave it in place
"""
import gzip
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse

import tfm_pipeline as tp

rng = np.random.default_rng(0)
LM22_TYPES = [f"type_{i}" for i in range(22)]
LM22_GENES = [f"LMG{i}" for i in range(120)]


def expression(age, genes, signal_genes, scale=1.0):
    z = (age - age.mean()) / age.std()
    X = rng.normal(5, 1, size=(len(genes), len(age)))
    for g in signal_genes:
        X[genes.index(g)] += signal_genes[g] * z
    return np.clip(X * scale, 0, None)


def write_gtex(d, signal):
    ages = rng.choice([25, 35, 45, 55, 65, 75], 120)
    samples = [f"GTEX-{i:05d}-0005-SM-X" for i in range(120)]
    genes = tp.PANEL_106 + LM22_GENES + ["OTHER1", "OTHER2"]
    X = expression(ages, genes, signal)
    rows = [f"ENSG{i:011d}.{rng.integers(1, 9)}\t{g}\t" + "\t".join(f"{v:.3f}" for v in X[i])
            for i, g in enumerate(genes)]
    rows.append(rows[genes.index("JUN")])                               # duplicated symbol
    with gzip.open(os.path.join(d, "gtex.gct.gz"), "wt") as fh:
        fh.write("#1.2\n" + f"{len(rows)}\t{len(samples)}\n" + "Name\tDescription\t" + "\t".join(samples) + "\n")
        fh.write("\n".join(rows) + "\n")
    ph = pd.DataFrame({"SUBJID": [s[:10] for s in samples], "SEX": rng.choice([1, 2], 120),
                       "AGE": [f"{a - 5}-{a + 4}" for a in ages], "DTHHRDY": 0})
    ph.to_csv(os.path.join(d, "phenotypes.txt"), sep="\t", index=False)
    lm22 = pd.DataFrame(rng.gamma(2, 50, size=(len(LM22_GENES), 22)), index=LM22_GENES, columns=LM22_TYPES)
    lm22.index.name = "Gene symbol"
    lm22.to_csv(os.path.join(d, "LM22.txt"), sep="\t")


def write_h5ad(path, n_donors, ages, signal, with_sex, with_raw, cells_per_donor=15):
    genes = tp.PANEL_106 + ["OTHER1"] + LM22_GENES[:10]
    donor_ids = np.repeat([f"D{i}" for i in range(n_donors)], cells_per_donor)
    cell_age = np.repeat(ages, cells_per_donor)
    mean = expression(cell_age, genes, signal, scale=0.4).T
    counts = sparse.csr_matrix(rng.poisson(np.exp(mean)).astype(np.float32))
    obs = pd.DataFrame({"donor_id": pd.Categorical(donor_ids), "age": cell_age,
                        "cell_type": pd.Categorical(rng.choice(["T cell", "B cell", "NK", "mono"], len(donor_ids)))},
                       index=[f"cell{i}" for i in range(len(donor_ids))])
    if with_sex:
        sex = dict(zip(np.unique(donor_ids), rng.choice(["male", "female"], n_donors)))
        obs["sex"] = pd.Categorical([sex[d] for d in donor_ids])
    var = pd.DataFrame({"feature_name": pd.Categorical(genes)}, index=[f"ENSG{i:011d}" for i in range(len(genes))])
    a = ad.AnnData(X=counts, obs=obs, var=var)
    if with_raw:
        a.raw = a.copy()
        a.X = sparse.csr_matrix(np.log1p(counts.toarray()))
    a.write_h5ad(path)


def main():
    full, keep = "--full" in sys.argv, "--keep" in sys.argv
    d = tempfile.mkdtemp(prefix="tfm_smoke_")
    signal = {g: b for g, b in zip(tp.PANEL_106[::2], rng.normal(0, 0.6, len(tp.PANEL_106[::2])))}
    write_gtex(d, signal)
    write_h5ad(os.path.join(d, "gse233321.h5ad"), 60, rng.uniform(0.2, 100, 60), signal, with_sex=False, with_raw=False)
    write_h5ad(os.path.join(d, "onek1k.h5ad"), 150, rng.integers(19, 98, 150).astype(float), signal, with_sex=True, with_raw=True)
    cfg = {"gtex_gct": os.path.join(d, "gtex.gct.gz"), "gtex_phenotypes": os.path.join(d, "phenotypes.txt"),
           "lm22": os.path.join(d, "LM22.txt"),
           "gse233321": {"h5ad": os.path.join(d, "gse233321.h5ad"), "chunk_cells": 200},
           "onek1k": {"h5ad": os.path.join(d, "onek1k.h5ad"), "obs": {"donor": "donor_id", "age": "age"}}}
    with open(os.path.join(d, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    argv = ["--config", os.path.join(d, "config.json"), "--out", os.path.join(d, "results")]
    if not full:
        argv += ["--skip-rf-sweep", "--skip-stepwise"]
    tp.main(argv)
    tables = sorted(os.listdir(os.path.join(d, "results", "tables")))
    figures = sorted(os.listdir(os.path.join(d, "results", "figures")))
    print(f"{len(tables)} tables, {len(figures)} figures")
    if keep:
        print(f"outputs kept in {d}/results")


if __name__ == "__main__":
    main()
