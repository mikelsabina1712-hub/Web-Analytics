"""Offline tests for correlations, tests and network objects."""
import networkx as nx
import numpy as np
import pandas as pd

from sp500net import networks as N

rng = np.random.default_rng(1)


def _data(t=300, n=12):
    f = rng.normal(0, 1, (t, 1))
    x = 0.6 * f + rng.normal(0, 1, (t, n))
    x[:, :4] += rng.normal(0, 1, (t, 1))      # a "sector" of 4 stocks
    return x


def test_pairwise_corr_matches_pandas_with_missing():
    x = _data()
    x[5:9, 2] = np.nan
    x[100, 7] = np.nan
    c, n = N.pairwise_corr(x)
    ref = pd.DataFrame(x).corr().to_numpy()
    assert np.allclose(c, ref, atol=1e-12)
    assert n[2, 7] == 300 - 5 and n[0, 1] == 300


def test_bonferroni_cutoff_matches_design():
    m = 468 * 467 // 2
    assert abs(N.implied_r(N.bonferroni_z(0.05, m), 502) - 0.22) < 0.01
    assert abs(N.implied_r(N.bonferroni_z(0.05, m), 251) - 0.31) < 0.01


def test_significance_bh_admits_at_least_bonferroni():
    c, n = N.pairwise_corr(_data())
    b = N.significant(c, n, 0.05, "bonferroni")
    h = N.significant(c, n, 0.05, "bh")
    assert (b <= h).all() and (b == b.T).all() and not b.diagonal().any()
    assert b[:4, :4][np.triu_indices(4, 1)].all()       # the sector pairs are found


def test_mst_matches_networkx():
    c, _ = N.pairwise_corr(_data())
    nodes = [f"S{i}" for i in range(c.shape[0])]
    e = N.mst_edges(c, nodes)
    assert len(e) == len(nodes) - 1
    full = N.to_graph(N.edges_from_mask(c, ~np.eye(len(nodes), dtype=bool), nodes), nodes)
    ref = nx.minimum_spanning_tree(full, weight="distance")
    assert np.isclose(e["d"].sum(), ref.size(weight="distance"))


def test_eigen_diagnostic_noise_vs_factor():
    noise = rng.normal(0, 1, (500, 100))
    assert N.eigen_diagnostic(noise)["n_above_mp"] == 0
    factor = noise + rng.normal(0, 1, (500, 1))
    d = N.eigen_diagnostic(factor)
    assert d["n_above_mp"] >= 1 and d["lambda1_share"] > 0.4


def test_distance_range():
    assert np.allclose(N.distance(np.array([1.0, 0.0, -1.0])), [0, np.sqrt(2), 2])
