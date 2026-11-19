"""Helpers for the S&P 500 co-movement network project.

Modules map to pipeline stages:
  config   parameters and paths
  http     GET with retries (APIs block)
  wiki     Wikipedia revisions and constituents table
  tickers  ticker formats and share classes
  prices   yfinance download and cache
  quality  sanity checks and Tiingo validation
"""
