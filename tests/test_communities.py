"""Offline tests: NMI/ARI against scikit-learn, modularity against NetworkX."""
import networkx as nx
import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from sp500net import centrality as Ce, communities as Co

rng = np.random.default_rng(3)


@pytest.mark.parametrize("k1,k2", [(11, 11), (11, 25), (3, 7)])
def test_nmi_ari_match_sklearn(k1, k2):
    a = rng.integers(0, k1, 400)
    b = np.where(rng.random(400) < 0.6, a, rng.integers(0, k2, 400))
    assert np.isclose(Co.nmi(a, b), normalized_mutual_info_score(a, b))
    assert np.isclose(Co.ari(a, b), adjusted_rand_score(a, b))


def test_identical_partitions_and_relabelling():
    a = rng.integers(0, 5, 100)
    perm = np.array([3, 0, 4, 1, 2])[a]
    assert np.isclose(Co.nmi(a, perm), 1) and np.isclose(Co.ari(a, perm), 1)


def _graph():
    g = nx.planted_partition_graph(4, 15, 0.6, 0.05, seed=1)
    for u, v in g.edges():
        g[u][v]["weight"] = rng.uniform(0.2, 0.6)
    return g


def test_modularity_matches_networkx():
    g = _graph()
    labels = np.array([n // 15 for n in g.nodes()])
    u, v = np.array(list(g.edges())).T
    w = np.array([g[a][b]["weight"] for a, b in zip(u, v)])
    comms = [set(np.flatnonzero(labels == c)) for c in range(4)]
    for gamma in (1.0, 1.5):
        ref = nx.community.modularity(g, comms, weight="weight", resolution=gamma)
        assert np.isclose(Co.modularity(u, v, w, labels, gamma), ref)


def test_louvain_recovers_planted_partition_and_null_rejects():
    g = _graph()
    truth = np.array([n // 15 for n in g.nodes()])
    runs = Co.louvain_runs(g, range(5))
    parts = [np.array([r["partition"][n] for n in g.nodes()]) for r in runs]
    assert Co.ari(truth, parts[0]) > 0.9 and Co.mean_pairwise_ari(parts) > 0.9
    null = Co.permutation_null(truth, parts[0], 200, np.random.default_rng(0))
    s = Co.null_summary(Co.nmi(truth, parts[0]), null["nmi"])
    assert s["p"] == 1 / 201 and s["z"] > 10


def test_tree_betweenness_unweighted_equals_weighted():
    t = nx.random_labeled_tree(30, seed=2)
    for u, v in t.edges():
        t[u][v]["distance"] = rng.uniform(0.5, 1.5)
    ref = nx.betweenness_centrality(t, normalized=True, weight="distance")
    out = Ce.tree_centrality(t, "mst")["mst_betweenness"]
    assert np.allclose(out.sort_index().to_numpy(), [ref[n] for n in sorted(ref)])
