"""Ticker treatment and share classes (Issue 3)."""
from __future__ import annotations

import re

import pandas as pd


def to_yahoo(symbol: str) -> str:
    """Wikipedia/NYSE style to Yahoo/Tiingo style: ``BRK.B`` -> ``BRK-B``."""
    return symbol.strip().upper().replace(".", "-")


def _issuer_from_name(name: str) -> str:
    """Fallback issuer key when no CIK: drop class labels from the name."""
    name = re.sub(r"\(?\bclass\s+[a-z]\b\)?", "", name, flags=re.I)
    name = re.sub(r"\b(series\s+[a-z]|non-?voting|voting)\b", "", name, flags=re.I)
    return re.sub(r"[^a-z0-9]", "", name.lower())


def build_ticker_map(constituents: pd.DataFrame) -> pd.DataFrame:
    """One row per listed security with its Yahoo ticker and issuer.

    ``issuer_id`` is the SEC CIK when present (GOOG and GOOGL share one),
    otherwise a normalised company name. ``n_classes`` > 1 flags issuers
    whose classes must be reduced to one after prices are downloaded
    (rule: longest history, then highest dollar volume).
    """
    m = constituents.copy()
    m = m.rename(columns={"symbol": "wiki_symbol"})
    m["yahoo_symbol"] = m["wiki_symbol"].map(to_yahoo)
    m["issuer_id"] = m["cik"].where(m["cik"].ne(""), m["security"].map(_issuer_from_name))
    m["n_classes"] = m.groupby("issuer_id")["wiki_symbol"].transform("count")
    cols = ["wiki_symbol", "yahoo_symbol", "security", "issuer_id", "n_classes",
            "sector", "sub_industry", "cik", "date_added"]
    return m[cols].sort_values("wiki_symbol").reset_index(drop=True)
