"""Offline tests for the robustness specifications."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sp500net import networks as N, robustness as R

rng = np.random.default_rng(6)


def _returns(t=400, n=30):
    m = rng.normal(0, 0.01, t)
    f = rng.normal(0, 0.006, t)
    x = np.outer(m, rng.uniform(0.5, 1.5, n)) + rng.normal(0, 0.01, (t, n))
    x[:, :10] += f[:, None]
    return pd.DataFrame(x, columns=[f"S{i}" for i in range(n)])


def test_ew_loo_residuals_remove_the_market():
    r = _returns()
    e = R.ew_loo_residuals(r)
    loo = (r.sum(axis=1).to_numpy()[:, None] - r.to_numpy()) / (r.shape[1] - 1)
    for j in (0, 15):
        assert abs(np.corrcoef(e.iloc[:, j], loo[:, j])[0, 1]) < 1e-10
    c, _ = N.pairwise_corr(e.to_numpy())
    within = c[:10, :10][np.triu_indices(10, 1)].mean()
    others = c[10:, 10:][np.triu_indices(20, 1)].mean()
    assert within > others + 0.05 and abs(others) < 0.05


def test_spearman_input_matches_scipy():
    r = _returns()
    c, _ = N.pairwise_corr(R.spearman_input(r).to_numpy())
    assert np.isclose(c[0, 1], spearmanr(r.iloc[:, 0], r.iloc[:, 1])[0])


def test_fixed_density_keeps_k_strongest():
    r = _returns()
    nodes = list(r.columns)
    e = R.build_edges(r.to_numpy(), nodes, "fixed", 0.05, k_edges=25)
    c, _ = N.pairwise_corr(r.to_numpy())
    assert len(e) == 25 and np.isclose(e["rho"].min(), np.sort(c[np.triu_indices(30, 1)])[-25])


def test_evaluate_recovers_planted_group():
    r = _returns()
    nodes = list(r.columns)
    labels = np.array(["A"] * 10 + ["B"] * 20)
    out = R.run_spec_window(R.ew_loo_residuals(r).to_numpy(), nodes, labels, "bonferroni", 0.05, range(3))
    assert out["n_isolated"] >= 15 and out["nmi_median"] > 0.9
