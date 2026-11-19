"""Cleaning, returns and the fixed universe (Issues 1, 3, 4, 6).

Order of operations, each step a function below:
  1. align prices to the trading calendar (SPY's trading days)
  2. mask stale price runs (identical prices) as missing
  3. daily log returns
  4. set listed corporate-action days to missing
  5. coverage rule per stock over the sample
  6. one share class per issuer
No value is ever imputed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def align_to_calendar(prices_wide: pd.DataFrame, calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.Series]:
    """Reindex a date x ticker price table to ``calendar``.

    Returns the aligned table and, per ticker, how many of its dates were
    dropped because they are not trading days in the calendar (e.g. a
    when-issued day only one provider reports).
    """
    outside = prices_wide.loc[~prices_wide.index.isin(calendar)].notna().sum()
    return prices_wide.reindex(calendar), outside.rename("dates_outside_calendar")


def mask_stale_runs(prices: pd.DataFrame, min_run: int) -> tuple[pd.DataFrame, pd.Series]:
    """Set prices inside runs of >= ``min_run`` identical values to missing.

    The first price of a run is kept (it is a genuine close); the repeats
    are stale. Returns the masked table and the number of prices masked
    per ticker.
    """
    masked = prices.copy()
    counts = {}
    for col in prices.columns:
        s = prices[col]
        valid = s.notna()
        # A new run starts wherever the value changes or a gap interrupts it.
        new_run = (s != s.shift()) | ~valid | ~valid.shift(fill_value=False)
        run_id = new_run.cumsum()
        run_len = s.groupby(run_id).transform("size")
        pos_in_run = s.groupby(run_id).cumcount()
        stale = valid & (run_len >= min_run) & (pos_in_run > 0)
        masked.loc[stale, col] = np.nan
        counts[col] = int(stale.sum())
    return masked, pd.Series(counts, name="stale_prices_masked")


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """r_t = ln(P_t / P_{t-1}); missing if either price is missing."""
    return np.log(prices.where(prices > 0)).diff()


def apply_set_missing(returns: pd.DataFrame, entries: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    """Set listed (ticker, date) returns to missing. Reports what was applied."""
    out = returns.copy()
    log = []
    for e in entries:
        t, d = e["ticker"], pd.Timestamp(e["date"])
        applied = t in out.columns and d in out.index
        old = float(out.at[d, t]) if applied else np.nan
        if applied:
            out.at[d, t] = np.nan
        log.append({"ticker": t, "date": d.date().isoformat(), "applied": applied,
                    "return_removed": old, "reason": e.get("reason", "")})
    return out, log


def _longest_nan_run(s: pd.Series) -> int:
    isna = s.isna().to_numpy()
    best = run = 0
    for v in isna:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def coverage_table(returns: pd.DataFrame, min_coverage: float, max_gap: int) -> pd.DataFrame:
    """Per-stock coverage over the sample period (rows of ``returns``).

    coverage = share of sample days with a valid return; max_gap = longest
    stretch of consecutive missing returns. A stock passes if
    coverage >= min_coverage and max_gap <= max_gap.
    """
    n = len(returns)
    out = pd.DataFrame({
        "n_valid": returns.notna().sum(),
        "coverage": returns.notna().sum() / n,
        "max_gap": returns.apply(_longest_nan_run),
        "first_valid": returns.apply(pd.Series.first_valid_index),
    })
    out["passes_coverage"] = (out["coverage"] >= min_coverage) & (out["max_gap"] <= max_gap)
    return out


def dollar_volume(close: pd.DataFrame, volume: pd.DataFrame,
                  start: str | None = None, end: str | None = None) -> pd.Series:
    """Average daily dollar volume (unadjusted close x volume) over [start, end]."""
    dv = (close * volume).loc[start:end]
    return dv.mean().rename("avg_dollar_volume")


def choose_share_classes(candidates: pd.DataFrame) -> pd.DataFrame:
    """Keep one security per issuer: most valid returns, then highest dollar volume.

    ``candidates`` needs columns ``wiki_symbol, issuer_id, n_valid, avg_dollar_volume``.
    Returns it with a boolean ``keep_class`` column and the reason for each drop.
    """
    c = candidates.sort_values(["issuer_id", "n_valid", "avg_dollar_volume", "wiki_symbol"],
                               ascending=[True, False, False, True]).copy()
    c["keep_class"] = ~c.duplicated("issuer_id", keep="first")
    first = c.groupby("issuer_id")["wiki_symbol"].transform("first")
    c["class_reason"] = np.where(c["keep_class"], "", "other class of " + first + " kept")
    return c.sort_index()
