"""Centrality measures (Issues 11, 12).

  strength           residual network, sum of edge weights rho
  degree             raw MST (number of branches), unweighted
  betweenness        raw MST, normalized; on a tree it depends only on the
                     tree's shape, so weighted and unweighted coincide
  PageRank           residual network, only to show it is redundant with strength
"""
from __future__ import annotations

import networkx as nx
import pandas as pd


def tree_centrality(tree: nx.Graph, prefix: str) -> pd.DataFrame:
    return pd.DataFrame({f"{prefix}_degree": dict(tree.degree()),
                         f"{prefix}_betweenness": nx.betweenness_centrality(tree, normalized=True)})


def network_centrality(graph: nx.Graph) -> pd.DataFrame:
    return pd.DataFrame({"strength": dict(graph.degree(weight="weight")),
                         "degree": dict(graph.degree()),
                         "pagerank": nx.pagerank(graph, weight="weight")})
