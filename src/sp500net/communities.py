"""Communities, comparison with GICS and the label-permutation null
(Issues 13, 14, 15).

NMI and ARI are implemented from the contingency table (the formulas in the
research design) and checked against scikit-learn in the tests. Modularity is
computed from edge arrays so the 1,000-permutation null runs in seconds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def encode(labels) -> np.ndarray:
    """Map any labels to 0..K-1."""
    return np.unique(np.asarray(labels), return_inverse=True)[1]


def contingency(a, b) -> np.ndarray:
    """n_ij = number of items in group i of ``a`` and group j of ``b``."""
    a, b = encode(a), encode(b)
    ka, kb = a.max() + 1, b.max() + 1
    return np.bincount(a * kb + b, minlength=ka * kb).reshape(ka, kb)


def _entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def nmi(a, b) -> float:
    """Normalized mutual information, arithmetic mean of the entropies:
    NMI = I(U, V) / ((H(U) + H(V)) / 2)."""
    n_ij = contingency(a, b)
    n = n_ij.sum()
    a_i, b_j = n_ij.sum(axis=1), n_ij.sum(axis=0)
    nz = n_ij > 0
    mi = (n_ij[nz] / n * np.log(n * n_ij[nz] / np.outer(a_i, b_j)[nz])).sum()
    h = (_entropy(a_i) + _entropy(b_j)) / 2
    return float(mi / h) if h > 0 else 1.0


def ari(a, b) -> float:
    """Adjusted Rand index from the contingency table (Hubert & Arabie)."""
    n_ij = contingency(a, b)
    comb = lambda x: x * (x - 1) / 2
    n = n_ij.sum()
    s_ij = comb(n_ij).sum()
    s_a, s_b = comb(n_ij.sum(axis=1)).sum(), comb(n_ij.sum(axis=0)).sum()
    expected = s_a * s_b / comb(n)
    max_index = (s_a + s_b) / 2
    return float((s_ij - expected) / (max_index - expected)) if max_index != expected else 1.0


def modularity(u: np.ndarray, v: np.ndarray, w: np.ndarray, labels: np.ndarray,
               resolution: float = 1.0) -> float:
    """Weighted modularity of a partition.

    Q = sum_c [ L_c / W  -  gamma * (S_c / 2W)^2 ]
    with W the total edge weight, L_c the weight inside community c and S_c
    the total strength of its nodes. ``u``, ``v`` are node indices, ``labels``
    a community code per node.
    """
    labels = np.asarray(labels)
    total = w.sum()
    inside = w[labels[u] == labels[v]].sum()
    strength = np.bincount(u, w, minlength=len(labels)) + np.bincount(v, w, minlength=len(labels))
    s_c = np.bincount(encode(labels), strength)
    return float(inside / total - resolution * ((s_c / (2 * total)) ** 2).sum())


def louvain_runs(graph, seeds, resolution: float = 1.0) -> list[dict]:
    """``community.best_partition`` once per seed (python-louvain, as in the
    course). Returns a list of {seed, partition: dict node -> community}."""
    import community as community_louvain
    return [{"seed": int(s),
             "partition": community_louvain.best_partition(graph, weight="weight",
                                                           resolution=resolution, random_state=int(s))}
            for s in seeds]


def mean_pairwise_ari(partitions: list[np.ndarray]) -> float:
    """Stability of Louvain: average ARI over all pairs of runs."""
    k = len(partitions)
    vals = [ari(partitions[i], partitions[j]) for i in range(k) for j in range(i + 1, k)]
    return float(np.mean(vals)) if vals else np.nan


def permutation_null(gics: np.ndarray, partition: np.ndarray, n_perm: int, rng: np.random.Generator,
                     q_edges: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
                     gics_all: np.ndarray | None = None) -> pd.DataFrame:
    """Shuffle GICS labels across stocks ``n_perm`` times and recompute NMI,
    ARI (against ``partition``) and, if ``q_edges`` is given, the modularity
    of the shuffled GICS partition on the full graph (``gics_all``).

    Keeps the network, the partition and the sector sizes; destroys only
    the link between a stock's sector and its place in the network.
    """
    rows = []
    for _ in range(n_perm):
        g = rng.permutation(gics)
        row = {"nmi": nmi(g, partition), "ari": ari(g, partition)}
        if q_edges is not None:
            row["q_gics"] = modularity(*q_edges, rng.permutation(gics_all))
        rows.append(row)
    return pd.DataFrame(rows)


def null_summary(observed: float, null: pd.Series) -> dict:
    """Empirical p-value (1 + #null >= obs) / (1 + n) and z-score."""
    n = len(null)
    sd = null.std(ddof=1)
    return {"p": (1 + (null >= observed).sum()) / (1 + n),
            "z": (observed - null.mean()) / sd if sd > 0 else np.nan,
            "null_mean": null.mean(), "null_p95": null.quantile(0.95)}


def louvain_window(edges: pd.DataFrame, nodes: list[str], seeds, resolution: float = 1.0) -> pd.DataFrame:
    """All Louvain runs for one window's residual network, in long format
    (seed, wiki_symbol, community). Module-level so it can run in parallel."""
    from sp500net.networks import to_graph
    g = to_graph(edges, nodes)
    out = []
    for r in louvain_runs(g, seeds, resolution):
        out.append(pd.DataFrame({"seed": r["seed"], "wiki_symbol": nodes,
                                 "community": [r["partition"][n] for n in nodes]}))
    return pd.concat(out, ignore_index=True)
