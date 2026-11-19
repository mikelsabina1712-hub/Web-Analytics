"""Offline tests for the data-collection helpers (no network needed)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sp500net import prices, quality, tickers, wiki
from sp500net.http import get_json

FIX = Path(__file__).parent / "fixtures"
ALIASES = {"Telecommunication Services": "Communication Services"}


@pytest.fixture
def new():
    return wiki.parse_constituents((FIX / "wiki_2024.html").read_text())


@pytest.fixture
def old():
    return wiki.parse_constituents((FIX / "wiki_2018.html").read_text())


# ----------------------------------------------------------------- wiki

def test_parse_modern_layout(new):
    assert len(new) == 7
    assert list(new.columns) == ["symbol", "security", "sector", "sub_industry", "cik", "date_added"]
    assert new.loc[new.symbol == "V", "sector"].item() == "Financials"  # footnote removed
    assert new.loc[new.symbol == "AAPL", "cik"].item() == "320193"      # leading zeros removed


def test_parse_old_layout_without_table_id(old):
    assert len(old) == 7
    assert "FB" in set(old.symbol)
    assert old.loc[old.symbol == "T", "sector"].item() == "Telecommunication Services"


def test_crosscheck_agrees(new):
    html = (FIX / "wiki_2024.html").read_text()
    res = wiki.crosscheck_read_html(html, new)
    assert res["ok"], res


def test_normalise_sectors(old):
    out, unknown = wiki.normalise_sectors(old, ALIASES)
    assert unknown == []
    assert out.loc[out.symbol == "T", "sector"].item() == "Communication Services"


def test_sector_changes_counts_classes_and_renames(old, new):
    old_n, _ = wiki.normalise_sectors(old, ALIASES)
    ch = wiki.sector_changes(old_n, new)
    # GOOGL, GOOG (both classes count), FB->META via CIK, V. Not T: alias only.
    assert set(ch.symbol_after) == {"GOOGL", "GOOG", "META", "V"}
    assert ch.loc[ch.symbol_after == "META", "matched_on"].item() == "cik"


# -------------------------------------------------------------- tickers

def test_to_yahoo():
    assert tickers.to_yahoo("BRK.B") == "BRK-B"
    assert tickers.to_yahoo(" bf.b ") == "BF-B"


def test_ticker_map_flags_share_classes(new):
    m = tickers.build_ticker_map(new)
    assert m.loc[m.wiki_symbol == "GOOG", "n_classes"].item() == 2
    assert m.loc[m.wiki_symbol == "AAPL", "n_classes"].item() == 1
    assert m.loc[m.wiki_symbol == "BRK.B", "yahoo_symbol"].item() == "BRK-B"


def test_issuer_from_name_without_cik():
    assert tickers._issuer_from_name("Fox Corp (Class A)") == tickers._issuer_from_name("Fox Corp (Class B)")


# --------------------------------------------------------------- prices

def _fake_yf(tickers_, start, end):
    idx = pd.bdate_range("2015-01-02", periods=5)
    data = {}
    for t in tickers_:
        if t == "DEAD":
            continue
        base = np.arange(1, 6, dtype=float) * (1 + len(t))
        for f, v in (("Adj Close", base), ("Close", base * 1.1), ("Volume", base * 1e6)):
            data[(f, t)] = v
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["Price", "Ticker"])
    return df


def test_download_prices_caches_and_reports_failures(tmp_path):
    logs = []
    p, failed = prices.download_prices(["AAA", "BB", "DEAD"], "2015-01-01", "2015-02-01", tmp_path,
                                       batch_size=2, max_retries=2, downloader=_fake_yf,
                                       sleep=lambda s: None, log=logs.append)
    assert failed == ["DEAD"]
    assert set(p.ticker) == {"AAA", "BB"} and len(p) == 10
    # Second call must come entirely from cache.
    def boom(*a):
        raise AssertionError("should not download")
    p2, _ = prices.download_prices(["AAA", "BB", "DEAD"], "2015-01-01", "2015-02-01", tmp_path,
                                   batch_size=2, downloader=boom, log=logs.append)
    pd.testing.assert_frame_equal(p, p2)


def test_to_long_single_ticker_flat_columns():
    raw = _fake_yf(["AAA"], None, None).droplevel(1, axis=1)
    out = prices.to_long(raw, ["AAA"])
    assert set(out.columns) == {"date", "ticker", "adj_close", "close", "volume"}
    assert len(out) == 5


# -------------------------------------------------------------- quality

def test_sanity_checks_flags():
    d = pd.bdate_range("2015-01-02", periods=8)
    good = pd.DataFrame({"date": d, "ticker": "G", "adj_close": np.linspace(10, 11, 8)})
    stale = pd.DataFrame({"date": d, "ticker": "S", "adj_close": [5, 5, 5, 5, 5, 6, 7, 8.0]})
    jump = pd.DataFrame({"date": d, "ticker": "J", "adj_close": [10, 10.1, 30, 30.2, 30.1, 30, 30.3, 30.2]})
    out = quality.sanity_checks(pd.concat([good, stale, jump]), 0.5, 5).set_index("ticker")
    assert not out.loc["G", "any_flag"]
    assert out.loc["S", "flag_stale"] and out.loc["S", "longest_identical_run"] == 5
    assert out.loc["J", "flag_extreme"] and out.loc["J", "n_extreme_returns"] == 1


def test_compare_returns_ignores_level_differences():
    idx = pd.bdate_range("2015-01-02", periods=300)
    rng = np.random.default_rng(0)
    p = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))), index=idx)
    res = quality.compare_returns(p, p * 0.8, 0.001, 0.999, 0.01)  # different adjustment base
    assert res["passed"] and res["share_off"] == 0
    bad = p.copy()
    bad.iloc[150:] *= 1.05  # an unadjusted 5% jump
    assert quality.compare_returns(p, bad, 0.001, 0.999, 0.01)["max_abs_diff"] > 0.04


# ----------------------------------------------------------------- http

class _Resp:
    def __init__(self, status, payload=None, headers=None):
        self.status_code, self._p, self.headers = status, payload, headers or {}

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class _Session:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), 0

    def get(self, *a, **k):
        self.calls += 1
        return self.responses.pop(0)


def test_get_json_retries_on_429_then_succeeds():
    s = _Session([_Resp(429, headers={"Retry-After": "1"}), _Resp(200, {"ok": 1})])
    waits = []
    assert get_json("u", session=s, sleep=waits.append) == {"ok": 1}
    assert waits == [1.0] and s.calls == 2


def test_get_json_does_not_retry_401():
    s = _Session([_Resp(401), _Resp(200, {})])
    with pytest.raises(RuntimeError):
        get_json("u", session=s, sleep=lambda x: None)
    assert s.calls == 1
