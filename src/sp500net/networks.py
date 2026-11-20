"""Correlations, significance tests and the four network objects per window
(Issues 6, 8, 9, 10, 11).

  raw correlation matrix      complete, no threshold (connectivity)
  raw MST                     N-1 strongest links on distance d (connectivity, centrality)
  residual thresholded net    significant positive residual correlations, weight rho
  residual MST                robustness for centrality, figures

Every edge carries both attributes: weight = rho (similarity, for strength
and Louvain) and distance d = sqrt(2(1 - rho)) (for paths, MST, betweenness).
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree


def pairwise_corr(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise-complete Pearson correlation of the columns of ``x`` (T x N).

    Same result as ``pandas.DataFrame.corr()`` but computed with matrix
    products, so it is fast for 468 stocks. Returns (corr, n) where n[i, j]
    is the number of days on which both i and j have a return.
    """
    m = (~np.isnan(x)).astype(float)
    x0 = np.where(m > 0, x, 0.0)
    n = m.T @ m
    sx = x0.T @ m            # sx[i, j] = sum of x_i over days where j is also valid
    sxx = (x0 ** 2).T @ m
    sxy = x0.T @ x0
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy - sx * sx.T / n
        var_i = sxx - sx ** 2 / n
        corr = cov / np.sqrt(var_i * var_i.T)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0), n.astype(int)


def distance(rho: np.ndarray) -> np.ndarray:
    """d = sqrt(2(1 - rho)): 0 for rho = 1, 2 for rho = -1."""
    return np.sqrt(np.clip(2.0 * (1.0 - rho), 0.0, None))


def fisher_z(rho: np.ndarray, n: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Test statistic for H0: rho = 0, approximately N(0, 1).

    ``scale`` is the variance factor of artanh(r): 1 for Pearson, 1.06 for
    Spearman rank correlation (Fieller, Hartley and Pearson, 1957).
    """
    r = np.clip(rho, -0.999999, 0.999999)
    return np.arctanh(r) * np.sqrt(np.maximum(n - 3, 1) / scale)


def bonferroni_z(alpha: float, n_tests: int) -> float:
    """Two-sided critical |z| for family-wise level ``alpha`` over ``n_tests``."""
    return NormalDist().inv_cdf(1 - alpha / (2 * n_tests))


def implied_r(z_crit: float, n_days: int) -> float:
    """Smallest |r| that is significant with ``n_days`` observations."""
    return float(np.tanh(z_crit / np.sqrt(n_days - 3)))


def two_sided_p(z: np.ndarray) -> np.ndarray:
    from scipy.special import ndtr
    return 2 * ndtr(-np.abs(z))


def significant(rho: np.ndarray, n: np.ndarray, alpha: float, method: str = "bonferroni",
                scale: float = 1.0) -> np.ndarray:
    """Boolean N x N matrix of significant pairs (either sign), diagonal False.

    method="bonferroni": |z| above the family-wise cutoff (main, Issue 9).
    method="bh": Benjamini-Hochberg at FDR ``alpha`` (robustness).
    """
    iu = np.triu_indices(rho.shape[0], k=1)
    z = fisher_z(rho[iu], n[iu], scale)
    m = len(z)
    if method == "bonferroni":
        sig = np.abs(z) > bonferroni_z(alpha, m)
    elif method == "bh":
        p = two_sided_p(z)
        order = np.argsort(p)
        passed = p[order] <= alpha * np.arange(1, m + 1) / m
        k = passed.nonzero()[0].max() + 1 if passed.any() else 0
        sig = np.zeros(m, bool)
        sig[order[:k]] = True
    else:
        raise ValueError(method)
    out = np.zeros(rho.shape, bool)
    out[iu] = sig
    return out | out.T


def edges_from_mask(rho: np.ndarray, mask: np.ndarray, nodes: list[str]) -> pd.DataFrame:
    """Edge list (upper triangle) of the pairs in ``mask``."""
    i, j = np.nonzero(np.triu(mask, k=1))
    r = rho[i, j]
    return pd.DataFrame({"u": np.asarray(nodes)[i], "v": np.asarray(nodes)[j],
                         "rho": r, "d": distance(r)})


def mst_edges(rho: np.ndarray, nodes: list[str]) -> pd.DataFrame:
    """Minimum spanning tree on d (= maximum spanning tree on rho), N-1 edges."""
    d = distance(rho)
    # csgraph treats exact zeros as missing edges; a tiny offset keeps every pair.
    tree = minimum_spanning_tree(np.triu(d + 1e-12, k=1)).tocoo()
    i, j = np.minimum(tree.row, tree.col), np.maximum(tree.row, tree.col)
    if len(i) != len(nodes) - 1:
        raise RuntimeError("MST is not spanning: the correlation matrix has missing pairs")
    r = rho[i, j]
    return pd.DataFrame({"u": np.asarray(nodes)[i], "v": np.asarray(nodes)[j],
                         "rho": r, "d": distance(r)})


def eigen_diagnostic(x: np.ndarray) -> dict:
    """Marchenko-Pastur diagnostic on the listwise-complete days (Issue 8).

    lambda1_share = largest eigenvalue / N (share of variance in the common
    mode); n_above_mp = eigenvalues above the pure-noise bound (1 + sqrt(q))^2.
    """
    xl = x[~np.isnan(x).any(axis=1)]
    t, n = xl.shape
    ev = np.linalg.eigvalsh(np.corrcoef(xl, rowvar=False))
    q = n / t
    bound = (1 + np.sqrt(q)) ** 2
    return {"T_listwise": t, "q": q, "mp_bound": bound, "lambda1": ev[-1],
            "lambda1_share": ev[-1] / n, "n_above_mp": int((ev > bound).sum())}


def to_graph(edges: pd.DataFrame, nodes: list[str]):
    """NetworkX graph with every node (isolated ones too) and both edge attributes."""
    import networkx as nx
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from((u, v, {"weight": r, "distance": d})
                     for u, v, r, d in edges[["u", "v", "rho", "d"]].itertuples(index=False))
    return g
