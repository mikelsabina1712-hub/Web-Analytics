"""Null models for the snapshots (Issue 15).

  time shuffle   permute each stock's residual returns in time independently:
                 keeps every return distribution, destroys all
                 cross-correlation; a calibrated threshold then admits
                 (almost) no edges
  rewiring       degree-preserving double-edge swaps: keeps every stock's
                 degree, destroys who connects to whom; tests whether the
                 network is more modular than its degree sequence implies
"""
from __future__ import annotations

import networkx as nx
import numpy as np


def time_shuffle(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle each column of ``x`` (T x N) independently (missing values move too)."""
    return rng.permuted(x, axis=0)


def rewire(graph: nx.Graph, swaps_per_edge: int, seed: int) -> nx.Graph:
    """Unweighted copy of ``graph`` after swaps_per_edge * E double-edge swaps."""
    g = nx.Graph()
    g.add_nodes_from(graph.nodes())
    g.add_edges_from(graph.edges())
    e = g.number_of_edges()
    nx.double_edge_swap(g, nswap=swaps_per_edge * e, max_tries=100 * swaps_per_edge * e, seed=seed)
    return g


def best_unweighted_q(graph: nx.Graph, seeds) -> float:
    """Highest modularity over Louvain runs, ignoring edge weights."""
    import community as community_louvain
    best = -1.0
    for s in seeds:
        p = community_louvain.best_partition(graph, weight="__unweighted__", random_state=int(s))
        best = max(best, community_louvain.modularity(p, graph, weight="__unweighted__"))
    return best
