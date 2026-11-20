"""Alternative specifications for the robustness checks (Notebook 07).

Every check changes one choice of the main specification and reruns the
same steps: residuals -> correlations -> edges -> Louvain -> NMI/ARI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sp500net import communities as Co
from sp500net import networks as N

SPEARMAN_SCALE = 1.06


def ew_loo_residuals(r: pd.DataFrame) -> pd.DataFrame:
    """Market model with an equal-weighted leave-one-out market (Issue 7).

    For stock i the market on day t is the average return of all *other*
    stocks with a return that day, so a stock never explains itself. OLS
    alpha and beta per stock on its own market series, within the window.
    """
    x = r.to_numpy()
    valid = ~np.isnan(x)
    total = np.nansum(x, axis=1, keepdims=True)
    count = valid.sum(axis=1, keepdims=True)
    m = (total - np.where(valid, x, 0.0)) / (count - valid)     # leave-one-out mean
    m = np.where(valid, m, np.nan)
    dm, dx = m - np.nanmean(m, axis=0), x - np.nanmean(x, axis=0)
    beta = np.nansum(dm * dx, axis=0) / np.nansum(dm ** 2, axis=0)
    alpha = np.nanmean(x, axis=0) - beta * np.nanmean(m, axis=0)
    return pd.DataFrame(x - alpha - beta * m, index=r.index, columns=r.columns)


def spearman_input(r: pd.DataFrame) -> pd.DataFrame:
    """Ranks per stock; their Pearson correlation is the Spearman correlation
    (missing days are left out of the ranking)."""
    return r.rank()


def build_edges(x: np.ndarray, nodes: list[str], rule: str, alpha: float,
                k_edges: int | None = None, scale: float = 1.0) -> pd.DataFrame:
    """Positive edges of one window under a threshold rule.

    rule = "bonferroni" | "bh": significant positive correlations;
    rule = "fixed": the ``k_edges`` strongest positive correlations.
    """
    c, n = N.pairwise_corr(x)
    if rule in ("bonferroni", "bh"):
        mask = N.significant(c, n, alpha, rule, scale) & (c > 0)
    elif rule == "fixed":
        iu = np.triu_indices(len(nodes), 1)
        vals = c[iu]
        keep = np.argsort(vals)[::-1][:k_edges]
        keep = keep[vals[keep] > 0]
        mask = np.zeros(c.shape, bool)
        mask[iu[0][keep], iu[1][keep]] = True
    else:
        raise ValueError(rule)
    return N.edges_from_mask(c, mask, nodes)


def evaluate(edges: pd.DataFrame, nodes: list[str], labels: np.ndarray, seeds,
             resolution: float = 1.0) -> dict:
    """Louvain over ``seeds``; NMI/ARI vs ``labels`` (isolated stocks left out)."""
    runs = Co.louvain_window(edges, nodes, seeds, resolution)
    idx = {s: i for i, s in enumerate(nodes)}
    u, v = edges["u"].map(idx).to_numpy(), edges["v"].map(idx).to_numpy()
    w = edges["rho"].to_numpy()
    connected = (np.bincount(u, minlength=len(nodes)) + np.bincount(v, minlength=len(nodes))) > 0
    rows = []
    for _, g in runs.groupby("seed"):
        p = g["community"].to_numpy()
        rows.append({"Q": Co.modularity(u, v, w, p),
                     "nmi": Co.nmi(labels[connected], p[connected]),
                     "ari": Co.ari(labels[connected], p[connected]),
                     "n_communities": int((np.bincount(Co.encode(p[connected])) >= 2).sum())})
    rr = pd.DataFrame(rows)
    best = rr["Q"].idxmax()
    return {"n_nodes": len(nodes), "n_edges": len(edges), "n_isolated": int((~connected).sum()),
            "Q_best": rr.loc[best, "Q"], "n_communities": int(rr.loc[best, "n_communities"]),
            "nmi_median": rr["nmi"].median(), "ari_median": rr["ari"].median(),
            "nmi_best": rr.loc[best, "nmi"], "ari_best": rr.loc[best, "ari"],
            "Q_gics": Co.modularity(u, v, w, labels)}


def run_spec_window(x: np.ndarray, nodes: list[str], labels: np.ndarray, rule: str, alpha: float,
                    seeds, k_edges: int | None = None, scale: float = 1.0) -> dict:
    """One specification on one window (module-level, so it runs in parallel)."""
    return evaluate(build_edges(x, nodes, rule, alpha, k_edges, scale), nodes, labels, seeds)
