"""Market model per window (Issue 7).

For each stock i in a window:  r_i,t = alpha_i + beta_i * r_m,t + e_i,t
estimated by OLS on the days where r_i is observed (the market is never
missing). With one regressor OLS is
    beta  = cov(r_i, r_m) / var(r_m)
    alpha = mean(r_i) - beta * mean(r_m)
computed on the same days. The residuals e are the "residual returns".
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def fit_market_model(returns: pd.DataFrame, market: pd.Series) -> pd.DataFrame:
    """OLS alpha, beta and R^2 for every column of ``returns`` on ``market``.

    Both inputs must cover the same window; missing stock returns are skipped
    pairwise (the market's mean and variance use the stock's valid days).
    """
    if market.isna().any():
        raise ValueError("Market returns have missing values")
    m = market.reindex(returns.index).to_numpy()[:, None]
    r = returns.to_numpy()
    valid = ~np.isnan(r)
    n = valid.sum(axis=0)
    mv = np.where(valid, m, np.nan)
    m_mean, r_mean = np.nanmean(mv, axis=0), np.nanmean(r, axis=0)
    dm, dr = mv - m_mean, r - r_mean
    var_m = np.nansum(dm ** 2, axis=0)
    cov = np.nansum(dm * dr, axis=0)
    beta = cov / var_m
    alpha = r_mean - beta * m_mean
    ss_tot = np.nansum(dr ** 2, axis=0)
    r2 = cov ** 2 / (var_m * ss_tot)
    return pd.DataFrame({"alpha": alpha, "beta": beta, "r2": r2, "n_obs": n}, index=returns.columns)


def residual_returns(returns: pd.DataFrame, market: pd.Series, params: pd.DataFrame) -> pd.DataFrame:
    """e_i,t = r_i,t - alpha_i - beta_i * r_m,t (missing stays missing)."""
    p = params.reindex(returns.columns)
    m = market.reindex(returns.index).to_numpy()[:, None]
    e = returns.to_numpy() - p["alpha"].to_numpy() - p["beta"].to_numpy() * m
    return pd.DataFrame(e, index=returns.index, columns=returns.columns)
