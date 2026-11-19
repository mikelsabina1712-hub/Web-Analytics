"""Daily prices from Yahoo Finance via yfinance (Issue 4).

Downloads run in batches and every batch is cached on disk, so a Colab
disconnect never forces a full re-download. Prices are stored exactly as
received, in long format (date, ticker, adj_close, close, volume).

``adj_close`` (split- and dividend-adjusted) feeds the returns.
``close`` x ``volume`` gives dollar volume for share-class and top-100
choices, which needs the unadjusted price.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

FIELDS = {"Adj Close": "adj_close", "Close": "close", "Volume": "volume"}


def _yf_download(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf  # imported lazily so tests run without it
    return yf.download(
        tickers=tickers, start=start, end=end, interval="1d",
        auto_adjust=False, actions=False, group_by="column",
        threads=True, progress=False,
    )


def to_long(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Turn yfinance's wide output into long format.

    Handles both column layouts yfinance produces: a (field, ticker)
    MultiIndex, or flat field columns for a single ticker.
    """
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "ticker", *FIELDS.values()])
    frames = []
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = set(raw.columns.get_level_values(0))
        field_level = 0 if lvl0 & FIELDS.keys() else 1
        for t in tickers:
            try:
                sub = raw.xs(t, axis=1, level=1 - field_level)
            except KeyError:
                continue
            frames.append(sub.assign(ticker=t))
    else:
        frames.append(raw.assign(ticker=tickers[0]))
    df = pd.concat(frames)
    df = df.rename(columns=FIELDS)[["ticker", *FIELDS.values()]]
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df.rename_axis("date").reset_index()
    return df.dropna(subset=["adj_close"]).reset_index(drop=True)


def _batch_path(cache_dir: Path, tickers: list[str], start: str, end: str) -> Path:
    key = hashlib.sha1(json.dumps([sorted(tickers), start, end]).encode()).hexdigest()[:12]
    return cache_dir / f"batch_{key}.parquet"


def download_prices(tickers: list[str], start: str, end: str, cache_dir: Path,
                    batch_size: int = 50, max_retries: int = 4,
                    downloader=_yf_download, sleep=time.sleep, log=print) -> tuple[pd.DataFrame, list[str]]:
    """Download (or load from cache) daily prices for ``tickers``.

    Returns ``(long_prices, failed_tickers)``. A ticker fails if, after
    ``max_retries`` single-ticker attempts, it still has no data.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tickers = sorted(dict.fromkeys(tickers))
    parts = []
    batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]
    for n, batch in enumerate(batches, 1):
        path = _batch_path(cache_dir, batch, start, end)
        if path.exists():
            parts.append(pd.read_parquet(path))
            log(f"batch {n}/{len(batches)}: cached")
            continue
        try:
            df = to_long(downloader(batch, start, end), batch)
        except Exception as exc:  # network error: retry tickers one by one below
            log(f"batch {n}/{len(batches)}: failed ({exc}); retrying tickers individually")
            df = to_long(None, batch)
        missing = [t for t in batch if t not in set(df["ticker"])]
        for t in missing:
            for attempt in range(max_retries):
                sleep(2 ** (attempt + 1))
                try:
                    one = to_long(downloader([t], start, end), [t])
                except Exception:
                    continue
                if not one.empty:
                    df = pd.concat([df, one], ignore_index=True)
                    break
        df.to_parquet(path, index=False)
        got = df["ticker"].nunique()
        log(f"batch {n}/{len(batches)}: {got}/{len(batch)} tickers")
        parts.append(df)
    prices = pd.concat(parts, ignore_index=True) if parts else to_long(None, [])
    failed = sorted(set(tickers) - set(prices["ticker"]))
    return prices.sort_values(["ticker", "date"]).reset_index(drop=True), failed


def download_metadata(source: str, n_requested: int, failed: list[str], **extra) -> dict:
    """Provenance record saved next to every download (Issue 4: retrieval date)."""
    meta = {
        "source": source,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_requested": n_requested,
        "n_failed": len(failed),
        "failed": failed,
    }
    try:
        import yfinance
        meta["yfinance_version"] = yfinance.__version__
    except ImportError:
        pass
    meta.update(extra)
    return meta


def wide(prices: pd.DataFrame, field: str = "adj_close") -> pd.DataFrame:
    """Date x ticker table of one field."""
    return prices.pivot(index="date", columns="ticker", values=field).sort_index()
