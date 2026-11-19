"""Data-quality checks (Issue 4): sanity rules and independent validation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .http import get_json


# ------------------------------------------------------------ sanity checks

def _longest_identical_run(values: pd.Series) -> int:
    """Longest stretch of consecutive identical prices (1 = never repeats)."""
    v = values.to_numpy()
    if len(v) == 0:
        return 0
    best = run = 1
    for prev, cur in zip(v[:-1], v[1:]):
        run = run + 1 if cur == prev else 1
        best = max(best, run)
    return best


def sanity_checks(prices: pd.DataFrame, max_abs_return: float, max_identical_run: int) -> pd.DataFrame:
    """One row per ticker with the checks from the research design.

    Flags (not automatic exclusions; each is inspected by hand):
      * non-positive prices,
      * any |daily log return| above ``max_abs_return``,
      * a run of at least ``max_identical_run`` identical prices (stale data).
    """
    rows = []
    for t, g in prices.sort_values("date").groupby("ticker"):
        p = g["adj_close"]
        r = np.log(p.where(p > 0)).diff()
        extreme = g.loc[r.abs() > max_abs_return, "date"]
        rows.append({
            "ticker": t,
            "n_obs": len(g),
            "first_date": g["date"].min(),
            "last_date": g["date"].max(),
            "n_nonpositive": int((p <= 0).sum()),
            "n_extreme_returns": len(extreme),
            "extreme_dates": ";".join(d.strftime("%Y-%m-%d") for d in extreme),
            "longest_identical_run": _longest_identical_run(p),
        })
    out = pd.DataFrame(rows)
    out["flag_nonpositive"] = out["n_nonpositive"] > 0
    out["flag_extreme"] = out["n_extreme_returns"] > 0
    out["flag_stale"] = out["longest_identical_run"] >= max_identical_run
    out["any_flag"] = out[["flag_nonpositive", "flag_extreme", "flag_stale"]].any(axis=1)
    return out


# ------------------------------------------------------- Tiingo validation

def fetch_tiingo(ticker: str, start: str, end: str, api_key: str, cfg: dict, session=None) -> pd.Series:
    """Adjusted close from Tiingo's end-of-day REST endpoint.

    The key travels in the Authorization header, never in the URL, so it
    cannot leak into logs or browser history.
    """
    v = cfg["validation"]
    js = get_json(
        v["tiingo_url"].format(ticker=ticker),
        params={"startDate": start, "endDate": end, "format": "json"},
        headers={"Authorization": f"Token {api_key}", "Content-Type": "application/json"},
        max_retries=cfg["prices"]["max_retries"],
        session=session,
    )
    if not js:
        return pd.Series(dtype=float, name=ticker)
    df = pd.DataFrame(js)
    idx = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return pd.Series(df["adjClose"].to_numpy(dtype=float), index=idx, name=ticker).sort_index()


def compare_returns(a: pd.Series, b: pd.Series, day_tolerance: float,
                    min_corr: float, max_share_off: float) -> dict:
    """Compare daily log returns of the same stock from two sources.

    Returns, not prices: providers adjust past prices to different bases,
    so levels differ while returns should not.
    """
    ra = np.log(a.where(a > 0)).diff()
    rb = np.log(b.where(b > 0)).diff()
    both = pd.concat([ra.rename("a"), rb.rename("b")], axis=1, join="inner").dropna()
    if len(both) < 2:
        return {"n_days": len(both), "corr": np.nan, "share_off": np.nan,
                "max_abs_diff": np.nan, "passed": False}
    diff = (both["a"] - both["b"]).abs()
    corr = float(both["a"].corr(both["b"]))
    share_off = float((diff > day_tolerance).mean())
    return {
        "n_days": int(len(both)),
        "days_only_in_a": int(len(a.index.difference(b.index))),
        "days_only_in_b": int(len(b.index.difference(a.index))),
        "corr": corr,
        "share_off": share_off,
        "max_abs_diff": float(diff.max()),
        "passed": bool(corr > min_corr and share_off < max_share_off),
    }
