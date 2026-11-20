"""Offline tests for the snapshot null models."""
import networkx as nx
import numpy as np

from sp500net import networks as N, nulls as Nu

rng = np.random.default_rng(4)


def test_time_shuffle_keeps_columns_and_kills_correlation():
    f = rng.normal(0, 1, (500, 1))
    x = f + rng.normal(0, 0.5, (500, 40))
    s = Nu.time_shuffle(x, np.random.default_rng(0))
    assert np.allclose(np.sort(s, axis=0), np.sort(x, axis=0))
    c, n = N.pairwise_corr(s)
    assert N.significant(c, n, 0.05).sum() == 0
    c0, n0 = N.pairwise_corr(x)
    assert N.significant(c0, n0, 0.05).all(where=~np.eye(40, dtype=bool))


def test_rewire_keeps_degrees_and_lowers_modularity():
    g = nx.planted_partition_graph(4, 25, 0.5, 0.02, seed=3)
    r = Nu.rewire(g, 10, seed=1)
    assert dict(r.degree()) == dict(g.degree()) and r.number_of_edges() == g.number_of_edges()
    assert Nu.best_unweighted_q(g, range(3)) > Nu.best_unweighted_q(r, range(3)) + 0.2
