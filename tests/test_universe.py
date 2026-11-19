"""Offline tests for cleaning, returns and the universe rules."""
import numpy as np
import pandas as pd

from sp500net import universe as U

D = pd.bdate_range("2015-01-01", periods=10)


def test_align_drops_and_counts_off_calendar_dates():
    p = pd.DataFrame({"A": range(10)}, index=D, dtype=float)
    cal = D.delete(3)
    aligned, outside = U.align_to_calendar(p, cal)
    assert len(aligned) == 9 and outside["A"] == 1


def test_mask_stale_runs_keeps_first_of_run():
    p = pd.DataFrame({"A": [1, 2, 3, 3, 3, 3, 3, 4, 5, 6.0],
                      "B": [1, 1, 1, 1, 2, 3, 4, 5, 6, 7.0]}, index=D)
    m, n = U.mask_stale_runs(p, 5)
    assert n["A"] == 4 and n["B"] == 0          # B's run of 4 is below the limit
    assert m["A"].iloc[2] == 3 and m["A"].iloc[3:7].isna().all()


def test_mask_stale_runs_not_bridged_by_gap():
    p = pd.DataFrame({"A": [3, 3, 3, np.nan, 3, 3, 3, 4, 5, 6.0]}, index=D)
    _, n = U.mask_stale_runs(p, 5)
    assert n["A"] == 0


def test_log_returns_missing_price_removes_both_adjacent_returns():
    p = pd.DataFrame({"A": [100, 101, np.nan, 103, 104, 105, 106, 107, 108, 109.0]}, index=D)
    r = U.log_returns(p)
    assert r["A"].iloc[[0, 2, 3]].isna().all()
    assert np.isclose(r["A"].iloc[1], np.log(101 / 100))


def test_apply_set_missing():
    r = pd.DataFrame({"A": np.full(10, 0.01)}, index=D)
    out, log = U.apply_set_missing(r, [{"ticker": "A", "date": D[4]}, {"ticker": "Z", "date": D[4]}])
    assert np.isnan(out.at[D[4], "A"]) and log[0]["applied"] and not log[1]["applied"]
    assert np.isclose(log[0]["return_removed"], 0.01)


def test_coverage_rule():
    r = pd.DataFrame({"full": np.full(200, 0.0), "gappy": np.full(200, 0.0), "late": np.full(200, 0.0)},
                     index=pd.bdate_range("2015-01-01", periods=200))
    r.iloc[50:57, 1] = np.nan        # 7-day gap, coverage 96.5%
    r.iloc[:1, 2] = np.nan           # one missing day: 99.5%, gap 1
    t = U.coverage_table(r, 0.99, 5)
    assert t.loc["full", "passes_coverage"] and t.loc["late", "passes_coverage"]
    assert not t.loc["gappy", "passes_coverage"] and t.loc["gappy", "max_gap"] == 7


def test_choose_share_classes():
    c = pd.DataFrame({"wiki_symbol": ["GOOG", "GOOGL", "NWS", "NWSA", "AAPL"],
                      "issuer_id": ["1", "1", "2", "2", "3"],
                      "n_valid": [2516, 2516, 2516, 2400, 2516],
                      "avg_dollar_volume": [1.0, 2.0, 9.0, 1.0, 5.0]})
    out = U.choose_share_classes(c).set_index("wiki_symbol")
    assert out.loc["GOOGL", "keep_class"] and not out.loc["GOOG", "keep_class"]  # tie -> dollar volume
    assert out.loc["NWS", "keep_class"] and not out.loc["NWSA", "keep_class"]    # longer history wins
    assert out.loc["AAPL", "keep_class"]
