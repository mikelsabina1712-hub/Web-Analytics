"""Offline tests for windows and the market model."""
import numpy as np
import pandas as pd

from sp500net import market as M, windows as W

DATES = pd.bdate_range("2015-01-02", "2024-12-31")


def test_rolling_windows_count_and_bounds():
    w = W.rolling_windows(DATES, 24, 1)
    assert len(w) == 97
    assert w.iloc[0]["start"] == pd.Timestamp("2015-01-02") and w.iloc[0]["label"] == "2016-12"
    assert w.iloc[-1]["end"] == pd.Timestamp("2024-12-31") and w.iloc[-1]["label"] == "2024-12"
    assert w.iloc[1]["start"] == pd.Timestamp("2015-02-02")


def test_year_and_event_windows():
    y = W.year_windows(DATES, [2020])
    assert y.iloc[0]["start"].year == 2020 and y.iloc[0]["end"] == pd.Timestamp("2020-12-31")
    e = W.event_windows(DATES, 63, 5)
    assert (e["n_days"] == 63).all() and e.iloc[-1]["end"] == DATES[-1]
    assert (e["end"].diff().dropna() > pd.Timedelta(0)).all()
    first_end = DATES.get_loc(e.iloc[0]["end"])
    assert 62 <= first_end < 67


def test_all_windows_ids_unique():
    w = W.all_windows(DATES, {"rolling_months": 24, "rolling_step_months": 1,
                              "snapshot_years": [2019, 2020, 2022],
                              "event_window_days": 63, "event_step_days": 5})
    assert set(w["layer"]) == {"rolling", "snapshot", "event"}
    assert {"R2016-12", "S2020"} <= set(w["window_id"])


def test_market_model_recovers_beta_and_handles_missing():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2019-01-01", periods=500)
    m = pd.Series(rng.normal(0, 0.01, 500), index=idx)
    r = pd.DataFrame({"A": 0.0002 + 1.5 * m + rng.normal(0, 0.005, 500),
                      "B": -0.0001 + 0.5 * m + rng.normal(0, 0.005, 500)})
    r.iloc[10:20, 1] = np.nan
    p = M.fit_market_model(r, m)
    assert abs(p.loc["A", "beta"] - 1.5) < 0.05 and abs(p.loc["B", "beta"] - 0.5) < 0.05
    assert p.loc["B", "n_obs"] == 490
    # Matches numpy OLS on the valid days.
    ok = r["B"].notna()
    b, a = np.polyfit(m[ok], r["B"][ok], 1)
    assert np.isclose(p.loc["B", "beta"], b) and np.isclose(p.loc["B", "alpha"], a)
    e = M.residual_returns(r, m, p)
    assert e["B"].iloc[10:20].isna().all()
    assert abs(np.corrcoef(e["A"], m)[0, 1]) < 1e-10
    assert np.isclose(p.loc["A", "r2"], np.corrcoef(r["A"], m)[0, 1] ** 2)
