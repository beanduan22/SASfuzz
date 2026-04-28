"""Stage 1: collect filtered correctness-relevant issues.

Two operating modes:

- Live (no --snapshot, GITHUB_TOKEN set): runs the full Search API queries, applies
  the symptom-keyword filter on issue titles+bodies, and dumps the results.
- Replay (--snapshot points to a prior dump from results/github_issue_counts_v2.json):
  reuses the already-filtered URL set so the count is reproducible without API calls.

Outputs (in RQ1/data):
    raw_search.json        — per-query counts plus all unique hits (lightweight)
    filtered_issues.json   — list of {url, repo, number, title?, body?, labels?}
    collect_summary.json   — top-level counts table
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rq1.github_api import QuerySpec, search_all


HERE = Path(__file__).resolve().parents[2]
CONFIGS = HERE / "configs"
DATA = HERE / "data"


def load_specs(path: Path) -> list[QuerySpec]:
    rows = json.loads(path.read_text())
    return [
        QuerySpec(
            name=row["name"],
            repo=row["repo"],
            labels=tuple(row["labels"]),
            extra_terms=tuple(row.get("extra_terms", [])),
        )
        for row in rows
    ]


def _text_of(issue: dict) -> str:
    return f"{issue.get('title', '')}\n{issue.get('body', '') or ''}".lower()


def _keyword_match(text: str, keywords: list[str]) -> bool:
    haystack = text.lower()
    return any(k.lower() in haystack for k in keywords)


def _split_url(url: str) -> tuple[str, int]:
    parts = url.rstrip("/").split("/")
    return f"{parts[-4]}/{parts[-3]}", int(parts[-1])


def run_live(start: str, end: str, queries_path: Path, keywords: list[str]) -> dict:
    specs = load_specs(queries_path)
    raw_hits = 0
    unique: dict[str, dict] = {}
    per_query: list[dict] = []

    for spec in specs:
        items = search_all(spec, start, end)
        raw_hits += len(items)
        for it in items:
            unique[it["html_url"]] = it
        per_query.append(
            {"name": spec.name, "repo": spec.repo, "labels": list(spec.labels), "count": len(items)}
        )

    filtered = {url: it for url, it in unique.items() if _keyword_match(_text_of(it), keywords)}
    return _persist(start, end, keywords, raw_hits, unique, filtered, per_query)


def run_replay(snapshot: Path, start: str, end: str, keywords: list[str]) -> dict:
    raw = json.loads(snapshot.read_text())
    per_query = raw["per_query"]
    raw_hits = sum(q["count"] for q in per_query)
    unique: dict[str, dict] = {}
    for url in raw.get("unique_urls", []):
        repo, number = _split_url(url)
        unique[url] = {"html_url": url, "repository_url": f"https://api.github.com/repos/{repo}", "number": number}
    filtered_urls = set(raw.get("filtered_urls", []))
    filtered: dict[str, dict] = {url: unique[url] for url in filtered_urls if url in unique}
    return _persist(start, end, keywords, raw_hits, unique, filtered, per_query)


def _persist(
    start: str,
    end: str,
    keywords: list[str],
    raw_hits: int,
    unique: dict[str, dict],
    filtered: dict[str, dict],
    per_query: list[dict],
) -> dict:
    DATA.mkdir(parents=True, exist_ok=True)

    (DATA / "raw_search.json").write_text(
        json.dumps(
            {
                "date_range": {"start": start, "end": end},
                "per_query": per_query,
                "unique_count": len(unique),
                "unique_urls": sorted(unique),
            },
            indent=2,
        )
    )

    out_filtered: list[dict] = []
    for url, it in filtered.items():
        repo, number = _split_url(url)
        out_filtered.append(
            {
                "url": url,
                "repo": repo,
                "number": number,
                "title": it.get("title"),
                "body": it.get("body"),
                "labels": [l["name"] for l in it.get("labels", []) if isinstance(l, dict)] or None,
                "state": it.get("state"),
                "created_at": it.get("created_at"),
                "closed_at": it.get("closed_at"),
            }
        )
    out_filtered.sort(key=lambda r: (r["repo"], r["number"]))
    (DATA / "filtered_issues.json").write_text(json.dumps(out_filtered, indent=2))

    summary = {
        "date_range": {"start": start, "end": end},
        "keywords": keywords,
        "raw_hits": raw_hits,
        "unique_hits": len(unique),
        "filtered_hits": len(filtered),
        "per_query": per_query,
        "by_repo": {
            "pytorch/pytorch": sum(1 for u in filtered if "pytorch/pytorch" in u),
            "tensorflow/tensorflow": sum(1 for u in filtered if "tensorflow/tensorflow" in u),
        },
    }
    (DATA / "collect_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _default_snapshot() -> Path | None:
    candidate = HERE.parent / "results" / "github_issue_counts_v2.json"
    return candidate if candidate.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ1 stage 1: collect filtered issues.")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2026-04-30")
    parser.add_argument("--queries", type=Path, default=CONFIGS / "queries.json")
    parser.add_argument("--keywords", type=Path, default=CONFIGS / "keywords.json")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="Replay from a v2 counts snapshot (results/github_issue_counts_v2.json). "
             "Defaults to that path when present.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Force a live API run even if a snapshot exists.",
    )
    args = parser.parse_args()

    keywords = json.loads(args.keywords.read_text())["correctness_keywords"]

    if args.live:
        summary = run_live(args.start, args.end, args.queries, keywords)
    else:
        snap = args.snapshot or _default_snapshot()
        if snap is None:
            summary = run_live(args.start, args.end, args.queries, keywords)
        else:
            summary = run_replay(snap, args.start, args.end, keywords)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
