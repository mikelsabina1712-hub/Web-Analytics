"""Wikipedia: pinned revisions of "List of S&P 500 companies" (Issues 1, 2, 5).

Retrieval goes through the MediaWiki API (the course's ethics slide: use
an API when one exists). Parsing the returned HTML uses BeautifulSoup,
as in Lab 1.1, and is cross-checked against ``pandas.read_html``.
"""
from __future__ import annotations

import re
from io import StringIO

import pandas as pd
from bs4 import BeautifulSoup

from .http import get_json

# The 11 GICS sectors in use throughout 2016-2024.
GICS_SECTORS = {
    "Communication Services", "Consumer Discretionary", "Consumer Staples",
    "Energy", "Financials", "Health Care", "Industrials",
    "Information Technology", "Materials", "Real Estate", "Utilities",
}

# Column roles, matched on lower-cased header text. Order matters:
# the first pattern that matches a header wins.
_COLUMN_PATTERNS = [
    ("symbol", ("symbol", "ticker")),
    ("security", ("security", "company")),
    ("sub_industry", ("sub-industry", "sub industry", "subindustry")),
    ("sector", ("gics sector", "sector")),
    ("cik", ("cik",)),
    ("date_added", ("date added", "date first added", "added")),
]


# ---------------------------------------------------------------- API calls

def find_revision(cfg: dict, timestamp: str, session=None) -> dict:
    """Latest revision of the page at or before ``timestamp`` (ISO 8601, UTC).

    Returns ``{"revid", "timestamp", "comment"}``. Past revisions never
    change, so the same timestamp always resolves to the same revision.
    """
    w = cfg["wikipedia"]
    js = get_json(
        w["api_url"],
        params={
            "action": "query", "format": "json", "formatversion": 2,
            "prop": "revisions", "titles": w["page_title"],
            "rvlimit": 1, "rvdir": "older", "rvstart": timestamp,
            "rvprop": "ids|timestamp|comment",
        },
        headers={"User-Agent": w["user_agent"]},
        session=session,
    )
    page = js["query"]["pages"][0]
    if "missing" in page or not page.get("revisions"):
        raise ValueError(f"No revision of {w['page_title']!r} at or before {timestamp}")
    rev = page["revisions"][0]
    return {"revid": rev["revid"], "timestamp": rev["timestamp"], "comment": rev.get("comment", "")}


def fetch_revision_html(cfg: dict, revid: int, session=None) -> str:
    """Rendered HTML of one revision (``action=parse&oldid=...``)."""
    w = cfg["wikipedia"]
    js = get_json(
        w["api_url"],
        params={"action": "parse", "format": "json", "formatversion": 2,
                "oldid": revid, "prop": "text"},
        headers={"User-Agent": w["user_agent"]},
        session=session,
    )
    return js["parse"]["text"]


# ------------------------------------------------------------------ parsing

def _clean(text: str) -> str:
    """Strip footnote markers like [1] or [a] and surrounding whitespace."""
    return re.sub(r"\[[^\]]*\]", "", text).replace("\xa0", " ").strip()


def _map_headers(headers: list[str]) -> dict[str, int]:
    """Map each role (symbol, sector, ...) to a column index."""
    roles: dict[str, int] = {}
    for idx, raw in enumerate(headers):
        h = raw.lower()
        for role, patterns in _COLUMN_PATTERNS:
            if role in roles:
                continue
            if any(p in h for p in patterns):
                roles[role] = idx
                break
    return roles


def _find_constituents_table(soup: BeautifulSoup):
    """The constituents table: by id when present, else by its headers."""
    table = soup.find("table", id="constituents")
    if table is not None:
        return table
    for t in soup.find_all("table"):
        first_row = t.find("tr")
        if first_row is None:
            continue
        roles = _map_headers([_clean(c.get_text(" ")) for c in first_row.find_all(["th", "td"])])
        if {"symbol", "sector"} <= roles.keys():
            return t
    raise ValueError("No constituents table found in the HTML")


def parse_constituents(html: str) -> pd.DataFrame:
    """Parse the constituents table with BeautifulSoup.

    Returns one row per listed ticker with columns
    ``symbol, security, sector, sub_industry, cik, date_added``
    (``cik`` / ``date_added`` are empty if the revision lacks them).
    """
    soup = BeautifulSoup(html, "html.parser")
    table = _find_constituents_table(soup)
    rows = table.find_all("tr")
    headers = [_clean(c.get_text(" ")) for c in rows[0].find_all(["th", "td"])]
    roles = _map_headers(headers)
    missing = {"symbol", "security", "sector"} - roles.keys()
    if missing:
        raise ValueError(f"Constituents table lacks columns: {sorted(missing)} (headers: {headers})")

    records = []
    for tr in rows[1:]:
        cells = [_clean(c.get_text(" ")) for c in tr.find_all(["td", "th"])]
        if len(cells) < len(headers):
            continue  # spacer or malformed row
        rec = {role: cells[i] for role, i in roles.items()}
        if rec["symbol"]:
            records.append(rec)

    df = pd.DataFrame.from_records(records)
    for col in ("sub_industry", "cik", "date_added"):
        if col not in df:
            df[col] = ""
    df["cik"] = df["cik"].str.replace(r"\D", "", regex=True).str.lstrip("0")
    df["symbol"] = df["symbol"].str.upper().str.replace(r"\s+", "", regex=True)
    return df[["symbol", "security", "sector", "sub_industry", "cik", "date_added"]]


def crosscheck_read_html(html: str, parsed: pd.DataFrame) -> dict:
    """Compare the BeautifulSoup result with ``pandas.read_html``.

    Any disagreement means a parsing bug, not a data issue.
    """
    candidates = pd.read_html(StringIO(html))
    table = None
    for t in candidates:
        cols = [str(c[-1] if isinstance(c, tuple) else c) for c in t.columns]
        roles = _map_headers([_clean(c) for c in cols])
        if {"symbol", "sector"} <= roles.keys():
            t = t.copy()
            t.columns = cols
            table = t.rename(columns={cols[roles["symbol"]]: "symbol", cols[roles["sector"]]: "sector"})
            break
    if table is None:
        return {"ok": False, "reason": "read_html found no constituents table"}
    rh = table[["symbol", "sector"]].astype(str)
    rh["symbol"] = rh["symbol"].map(_clean).str.upper().str.replace(r"\s+", "", regex=True)
    rh["sector"] = rh["sector"].map(_clean)
    merged = parsed[["symbol", "sector"]].merge(rh, on="symbol", how="outer",
                                                 suffixes=("_bs4", "_pandas"), indicator=True)
    only_one = merged[merged["_merge"] != "both"]
    both = merged[merged["_merge"] == "both"]
    sector_diff = both[both["sector_bs4"] != both["sector_pandas"]]
    return {
        "ok": only_one.empty and sector_diff.empty,
        "n_bs4": len(parsed), "n_pandas": len(rh),
        "symbols_in_one_only": only_one["symbol"].tolist(),
        "sector_mismatches": sector_diff["symbol"].tolist(),
    }


def normalise_sectors(df: pd.DataFrame, aliases: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    """Apply sector aliases and return any labels outside the 11 GICS sectors."""
    out = df.copy()
    out["sector"] = out["sector"].replace(aliases)
    unknown = sorted(set(out["sector"]) - GICS_SECTORS)
    return out, unknown


# --------------------------------------------------- reclassification events

def match_tables(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Pair rows of two constituents tables.

    Each listed security is matched by symbol first; securities left
    unmatched (e.g. a ticker rename such as FB -> META) are then matched by
    CIK when both rows carry one. Share classes of one issuer stay separate
    rows, because index providers count each listed class as a constituent.
    """
    cols = ["symbol", "security", "sector", "cik"]
    b, a = before[cols].drop_duplicates("symbol"), after[cols].drop_duplicates("symbol")
    by_symbol = b.merge(a, on="symbol", suffixes=("_before", "_after"))
    by_symbol["symbol_before"] = by_symbol["symbol_after"] = by_symbol["symbol"]
    by_symbol["matched_on"] = "symbol"

    b_left = b[~b["symbol"].isin(by_symbol["symbol"]) & b["cik"].ne("")]
    a_left = a[~a["symbol"].isin(by_symbol["symbol"]) & a["cik"].ne("")]
    # Only CIKs with exactly one unmatched security on each side, so a
    # multi-class issuer is never paired with the wrong class.
    b_left = b_left[~b_left["cik"].duplicated(keep=False)]
    a_left = a_left[~a_left["cik"].duplicated(keep=False)]
    by_cik = b_left.merge(a_left, on="cik", suffixes=("_before", "_after"))
    by_cik["matched_on"] = "cik"

    keep = ["symbol_before", "symbol_after", "security_before", "sector_before",
            "sector_after", "matched_on"]
    if "cik" not in by_symbol:
        by_symbol["cik"] = by_symbol["cik_before"]
    return pd.concat([by_symbol[keep + ["cik"]], by_cik[keep + ["cik"]]], ignore_index=True)


def sector_changes(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Securities whose sector differs between two constituents tables.

    Only securities present in both tables count, which is how index
    providers report reclassification counts.
    """
    m = match_tables(before, after)
    return m[m["sector_before"] != m["sector_after"]].reset_index(drop=True)
