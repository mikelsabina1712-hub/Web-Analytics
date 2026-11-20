"""Estimation windows (Issue 16).

Three layers, all defined on the trading calendar of the return table:
  rolling   24 calendar months, stepped monthly; labelled by the last month
  snapshot  calendar years (2019, 2020, 2022)
  event     63 consecutive trading days, stepped every 5 trading days
Every window is a contiguous block of trading days, stored as start/end dates.
"""
from __future__ import annotations

import pandas as pd


def rolling_windows(dates: pd.DatetimeIndex, months: int, step: int = 1) -> pd.DataFrame:
    """Windows of ``months`` whole calendar months, ending at each month end.

    The first window starts in the first month of ``dates``; the last ends in
    its last month. With 2015-2024 and 24 months this gives 97 windows
    (ending Dec 2016 ... Dec 2024).
    """
    periods = dates.to_period("M")
    first, last = periods.min(), periods.max()
    rows = []
    end = first + (months - 1)
    while end <= last:
        start = end - (months - 1)
        inside = dates[(periods >= start) & (periods <= end)]
        rows.append({"layer": "rolling", "label": str(end), "start": inside.min(),
                     "end": inside.max(), "n_days": len(inside)})
        end += step
    return pd.DataFrame(rows)


def year_windows(dates: pd.DatetimeIndex, years: list[int]) -> pd.DataFrame:
    rows = []
    for y in years:
        inside = dates[dates.year == y]
        if len(inside) == 0:
            raise ValueError(f"No trading days in {y}")
        rows.append({"layer": "snapshot", "label": str(y), "start": inside.min(),
                     "end": inside.max(), "n_days": len(inside)})
    return pd.DataFrame(rows)


def event_windows(dates: pd.DatetimeIndex, length: int, step: int) -> pd.DataFrame:
    """Blocks of ``length`` trading days every ``step`` days, anchored so the
    last window ends on the last date. Labelled by the end date."""
    ends = list(range(len(dates) - 1, length - 2, -step))[::-1]
    return pd.DataFrame([{"layer": "event", "label": str(dates[e].date()),
                          "start": dates[e - length + 1], "end": dates[e], "n_days": length}
                         for e in ends])


def all_windows(dates: pd.DatetimeIndex, cfg_windows: dict) -> pd.DataFrame:
    w = pd.concat([
        rolling_windows(dates, cfg_windows["rolling_months"], cfg_windows["rolling_step_months"]),
        year_windows(dates, cfg_windows["snapshot_years"]),
        event_windows(dates, cfg_windows["event_window_days"], cfg_windows["event_step_days"]),
    ], ignore_index=True)
    w.insert(0, "window_id", [f"{l[0].upper()}{lab}" for l, lab in zip(w["layer"], w["label"])])
    assert w["window_id"].is_unique
    return w


def slice_window(frame: pd.DataFrame | pd.Series, window: pd.Series | dict):
    """Rows of ``frame`` inside one window (inclusive)."""
    return frame.loc[window["start"]:window["end"]]
