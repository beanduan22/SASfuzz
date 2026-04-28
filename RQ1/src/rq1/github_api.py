from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable

GITHUB_API = "https://api.github.com"


def _token() -> str | None:
    return os.environ.get("GITHUB_TOKEN") or None


def _headers(extra: dict | None = None) -> dict:
    h = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "sasfuzz-rq1",
    }
    tok = _token()
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    if extra:
        h.update(extra)
    return h


def _sleep_for_rate(token: str | None) -> None:
    if token:
        time.sleep(1.0)
    else:
        time.sleep(8.0)


def get_json(url: str, retries: int = 5) -> dict | list:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_headers())
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_err = exc
            if exc.code in (403, 429):
                wait = 30 * (attempt + 1)
                time.sleep(wait)
                continue
            if 500 <= exc.code < 600:
                time.sleep(5 * (attempt + 1))
                continue
            raise
        except urllib.error.URLError:
            time.sleep(5 * (attempt + 1))
            continue
    if last_err is not None:
        raise last_err
    raise RuntimeError("get_json: exhausted retries")


@dataclass(frozen=True)
class QuerySpec:
    name: str
    repo: str
    labels: tuple[str, ...]
    extra_terms: tuple[str, ...] = ()


def search_query(spec: QuerySpec, start: str, end: str) -> str:
    parts = [
        f"repo:{spec.repo}",
        "is:issue",
        "is:closed",
        f"created:{start}..{end}",
    ]
    parts.extend(f'label:"{label}"' for label in spec.labels)
    parts.extend(spec.extra_terms)
    return " ".join(parts)


def _search_page(query: str, page: int) -> dict:
    encoded = urllib.parse.quote(query)
    url = f"{GITHUB_API}/search/issues?q={encoded}&per_page=100&page={page}"
    return get_json(url)  # type: ignore[return-value]


def _parse(d: str) -> date:
    return datetime.strptime(d, "%Y-%m-%d").date()


def _fmt(d: date) -> str:
    return d.isoformat()


def search_all(spec: QuerySpec, start: str, end: str) -> list[dict]:
    """Return every result for the query, recursively splitting on the 1000-cap."""
    query = search_query(spec, start, end)
    first = _search_page(query, 1)
    total = int(first.get("total_count", 0))

    if total > 1000:
        s, e = _parse(start), _parse(end)
        if s >= e:
            raise ValueError(f"query exceeds 1000-cap on a single day: {query}")
        mid = s + (e - s) // 2
        left = search_all(spec, _fmt(s), _fmt(mid))
        right = search_all(spec, _fmt(mid + timedelta(days=1)), _fmt(e))
        return left + right

    items: list[dict] = list(first.get("items", []))
    page = 2
    while len(items) < total:
        _sleep_for_rate(_token())
        payload = _search_page(query, page)
        batch = payload.get("items", [])
        if not batch:
            break
        items.extend(batch)
        page += 1
        if len(batch) < 100:
            break
    return items


def fetch_issue_full(repo: str, number: int) -> dict:
    return get_json(f"{GITHUB_API}/repos/{repo}/issues/{number}")  # type: ignore[return-value]


def fetch_issue_events(repo: str, number: int) -> list[dict]:
    out: list[dict] = []
    page = 1
    while True:
        payload = get_json(
            f"{GITHUB_API}/repos/{repo}/issues/{number}/timeline?per_page=100&page={page}"
        )
        if not isinstance(payload, list) or not payload:
            break
        out.extend(payload)
        if len(payload) < 100:
            break
        page += 1
        _sleep_for_rate(_token())
    return out


def parse_repo_and_number(html_url: str) -> tuple[str, int]:
    parts = html_url.rstrip("/").split("/")
    return f"{parts[-4]}/{parts[-3]}", int(parts[-1])


def iter_filtered(items: Iterable[dict], keywords: list[str]) -> Iterable[dict]:
    keys = [k.lower() for k in keywords]
    for issue in items:
        text = f"{issue.get('title', '')}\n{issue.get('body', '') or ''}".lower()
        if any(k in text for k in keys):
            yield issue
