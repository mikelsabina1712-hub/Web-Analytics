"""HTTP GET with retries, as in the course's API block.

Retries only on errors that can go away by waiting: 429 (rate limit) and
5xx (server side). A 4xx such as 401 or 404 is raised immediately, because
retrying a wrong key or a missing ticker never helps.
"""
from __future__ import annotations

import time

import requests

RETRY_STATUS = {429, 500, 502, 503, 504}


def get_json(url: str, *, params: dict | None = None, headers: dict | None = None,
             max_retries: int = 4, timeout: int = 60, session: requests.Session | None = None,
             sleep=time.sleep) -> object:
    """GET ``url`` and return the decoded JSON body.

    Waits ``Retry-After`` seconds when the server sends it, otherwise
    2, 4, 8, ... seconds between attempts.
    """
    http = session or requests
    for attempt in range(max_retries + 1):
        try:
            r = http.get(url, params=params, headers=headers, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout):
            if attempt == max_retries:
                raise
            sleep(2 ** (attempt + 1))
            continue
        if r.status_code in RETRY_STATUS and attempt < max_retries:
            wait = r.headers.get("Retry-After")
            sleep(float(wait) if wait and wait.replace(".", "", 1).isdigit() else 2 ** (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("unreachable")  # loop always returns or raises
