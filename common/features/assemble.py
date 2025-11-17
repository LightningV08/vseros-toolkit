import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, issparse

from .store import FeatureStore


def make_dense(store: FeatureStore, include):
    dfs_tr, dfs_te, parts = [], [], []
    for name in include:
        pkg = store.get(name)
        assert pkg.kind == "dense", f"{name} not dense"
        dfs_tr.append(pkg.train)
        dfs_te.append(pkg.test)
        parts.append((name, len(pkg.cols)))
    Xtr = pd.concat(dfs_tr, axis=1)
    Xte = pd.concat(dfs_te, axis=1)
    print("[DENSE] parts:", parts, "→", Xtr.shape, Xte.shape)
    _validate_pair(Xtr, Xte)
    return Xtr, Xte, store.catalog()


def make_sparse(store: FeatureStore, include):
    mats_tr, mats_te, parts = [], [], []
    for name in include:
        pkg = store.get(name)
        assert pkg.kind == "sparse", f"{name} not sparse"
        mats_tr.append(csr_matrix(pkg.train))
        mats_te.append(csr_matrix(pkg.test))
        parts.append((name, "csr"))
    Xtr = hstack(mats_tr).tocsr()
    Xte = hstack(mats_te).tocsr()
    print("[SPARSE] parts:", parts, "→", Xtr.shape, Xte.shape)
    return Xtr, Xte, store.catalog()


def _validate_pair(Xtr, Xte):
    if issparse(Xtr):
        return
    assert list(Xtr.columns) == list(Xte.columns)
    assert np.isfinite(Xtr.to_numpy()).all()
    assert np.isfinite(Xte.to_numpy()).all()
