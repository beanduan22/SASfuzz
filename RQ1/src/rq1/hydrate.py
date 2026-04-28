"""Stage 2 (optional): fetch full title/body/labels for each filtered issue.

Only the fix-verification and classification stages need the full body. This stage
walks data/filtered_issues.json and fills in any missing 'title'/'body'/'labels'
fields by querying the GitHub Issues API. Results are cached per issue so reruns
are idempotent.

Cache layout:
    data/issue_cache/<repo>/<number>.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from rq1.github_api import fetch_issue_full


def _per_call_sleep() -> float:
    """Stay under unauthenticated 60 req/hr if no token; 1s otherwise."""
    return 1.0 if os.environ.get("GITHUB_TOKEN") else 65.0


HERE = Path(__file__).resolve().parents[2]
DATA = HERE / "data"
CACHE = DATA / "issue_cache"


def _cache_path(repo: str, number: int) -> Path:
    return CACHE / repo.replace("/", "__") / f"{number}.json"


def hydrate_one(repo: str, number: int) -> dict:
    cache = _cache_path(repo, number)
    if cache.exists():
        return json.loads(cache.read_text())
    payload = fetch_issue_full(repo, number)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(payload, indent=2))
    return payload


def _fix_verified_urls() -> set[str] | None:
    fv_path = DATA / "fix_verification.json"
    if not fv_path.exists():
        return None
    return {r["url"] for r in json.loads(fv_path.read_text()) if r["verdict"] == "fix_verified"}


def run(limit: int | None, only_fix_verified: bool = False) -> dict:
    src = json.loads((DATA / "filtered_issues.json").read_text())
    if only_fix_verified:
        urls = _fix_verified_urls()
        if urls is not None:
            src = [r for r in src if r["url"] in urls]

    fetched = 0
    cached = 0
    failed = 0
    out: list[dict] = []

    progress_path = DATA / "filtered_issues_full.json"
    existing: dict[str, dict] = {}
    if progress_path.exists():
        for r in json.loads(progress_path.read_text()):
            existing[r["url"]] = r
    out.extend(existing.get(r["url"], r) for r in [])  # placeholder

    def _flush() -> None:
        merged = list(existing.values())
        # add any rows already processed in this run
        seen = {m["url"] for m in merged}
        for r in out:
            if r["url"] not in seen:
                merged.append(r)
                seen.add(r["url"])
        progress_path.write_text(json.dumps(merged, indent=2))

    for row in src:
        if limit is not None and (fetched + cached) >= limit:
            break
        repo, number = row["repo"], row["number"]
        cache = _cache_path(repo, number)
        if not row.get("body") or not row.get("labels") or row.get("title") is None:
            try:
                if cache.exists():
                    payload = json.loads(cache.read_text())
                    cached += 1
                else:
                    payload = hydrate_one(repo, number)
                    fetched += 1
                    time.sleep(_per_call_sleep())
                row["title"] = payload.get("title", row.get("title"))
                row["body"] = payload.get("body", row.get("body"))
                row["labels"] = [
                    l["name"] for l in payload.get("labels", []) if isinstance(l, dict)
                ] or row.get("labels")
                row["state"] = payload.get("state", row.get("state"))
                row["closed_at"] = payload.get("closed_at", row.get("closed_at"))
                row["pull_request"] = payload.get("pull_request")
            except Exception as exc:
                failed += 1
                row["_error"] = str(exc)
        out.append(row)
        existing[row["url"]] = row
        if (fetched + cached) % 5 == 0:
            _flush()

    _flush()

    summary = {
        "total": len(out),
        "fetched_live": fetched,
        "served_from_cache": cached,
        "failed": failed,
    }
    (DATA / "hydrate_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ1 stage 2: hydrate issue bodies.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only-fix-verified", action="store_true",
                        help="Restrict to URLs marked fix_verified by stage 3.")
    args = parser.parse_args()
    print(json.dumps(run(args.limit, args.only_fix_verified), indent=2))


if __name__ == "__main__":
    main()
