"""Stage 3: classify each filtered issue as Fix-verified or Unverified.

Definition (matches the paper):
  An issue is Fix-verified iff its GitHub history provides a publicly auditable
  link to a merged fixing pull request OR a default-branch commit. We check four
  signals, in priority order:

    1. The issue itself was opened as a pull request (REST issues API surfaces a
       'pull_request' field). We *exclude* these — the paper's pool is issues only.
    2. The issue's timeline contains a 'closed' event whose 'commit_id' is non-null
       (the closing commit landed on the default branch).
    3. The timeline contains a 'cross-referenced' event whose source is a
       *merged* pull request.
    4. The body or any comment includes a 'Fixes #NNN' / 'Fixed by #NNN' reference
       to a pull request that is itself merged. (Optional, expensive — disabled by
       default; controlled via --deep.)

Inputs:
    data/filtered_issues_full.json (or filtered_issues.json if hydrate skipped)

Outputs:
    data/fix_verification.json     — per-issue verdict + signals
    data/fix_verification_summary.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

from empirical_study.github_api import QuerySpec, fetch_issue_events, get_json, search_all, GITHUB_API


HERE = Path(__file__).resolve().parents[2]
CONFIGS = HERE / "configs"
DATA = HERE / "data"
TIMELINE_CACHE = DATA / "timeline_cache"


CLOSE_REF_RE = re.compile(
    r"\b(close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s*[: ]*#(\d+)\b",
    re.IGNORECASE,
)


def _timeline_path(repo: str, number: int) -> Path:
    return TIMELINE_CACHE / repo.replace("/", "__") / f"{number}.json"


def load_timeline(repo: str, number: int) -> list[dict]:
    cache = _timeline_path(repo, number)
    if cache.exists():
        return json.loads(cache.read_text())
    events = fetch_issue_events(repo, number)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(events, indent=2))
    return events


def _pr_is_merged(repo: str, number: int) -> bool:
    try:
        payload = get_json(f"{GITHUB_API}/repos/{repo}/pulls/{number}")
    except Exception:
        return False
    return bool(payload.get("merged_at"))


def signals_for(issue: dict, deep: bool) -> dict:
    repo = issue["repo"]
    number = issue["number"]
    flags = {
        "is_pull_request": bool(issue.get("pull_request")),
        "closed_with_commit": False,
        "closed_with_merged_pr": False,
        "body_refs_merged_pr": False,
    }
    if flags["is_pull_request"]:
        return flags

    try:
        events = load_timeline(repo, number)
    except Exception as exc:
        flags["_timeline_error"] = str(exc)
        return flags

    for ev in events:
        kind = ev.get("event")
        if kind == "closed" and ev.get("commit_id"):
            flags["closed_with_commit"] = True
        if kind == "cross-referenced":
            src = ev.get("source", {})
            if src.get("type") == "issue":
                issue_obj = src.get("issue", {})
                if issue_obj.get("pull_request") and issue_obj.get("state") == "closed":
                    pr_url = issue_obj.get("pull_request", {}).get("html_url", "")
                    pr_repo, pr_num = repo, None
                    m = re.search(r"github\.com/([^/]+/[^/]+)/pull/(\d+)", pr_url)
                    if m:
                        pr_repo, pr_num = m.group(1), int(m.group(2))
                    if pr_num is None:
                        continue
                    if pr_repo == repo and _pr_is_merged(pr_repo, pr_num):
                        flags["closed_with_merged_pr"] = True

    if deep and not (flags["closed_with_commit"] or flags["closed_with_merged_pr"]):
        body = (issue.get("body") or "")
        for _verb, num in CLOSE_REF_RE.findall(body):
            try:
                if _pr_is_merged(repo, int(num)):
                    flags["body_refs_merged_pr"] = True
                    break
            except Exception:
                continue

    return flags


def verdict_of(flags: dict) -> str:
    if flags.get("is_pull_request"):
        return "excluded_pull_request"
    if (
        flags.get("closed_with_commit")
        or flags.get("closed_with_merged_pr")
        or flags.get("body_refs_merged_pr")
    ):
        return "fix_verified"
    return "unverified"


def run(deep: bool, limit: int | None) -> dict:
    src_path = DATA / "filtered_issues_full.json"
    if not src_path.exists():
        src_path = DATA / "filtered_issues.json"
    issues = json.loads(src_path.read_text())

    out: list[dict] = []
    counts = {"fix_verified": 0, "unverified": 0, "excluded_pull_request": 0}
    by_repo: dict[str, dict[str, int]] = {}

    for i, issue in enumerate(issues):
        if limit is not None and i >= limit:
            break
        flags = signals_for(issue, deep)
        v = verdict_of(flags)
        counts[v] = counts.get(v, 0) + 1
        bucket = by_repo.setdefault(issue["repo"], {"fix_verified": 0, "unverified": 0, "excluded_pull_request": 0})
        bucket[v] = bucket.get(v, 0) + 1

        out.append(
            {
                "url": issue["url"],
                "repo": issue["repo"],
                "number": issue["number"],
                "title": issue.get("title"),
                "verdict": v,
                "signals": flags,
            }
        )

    (DATA / "fix_verification.json").write_text(json.dumps(out, indent=2))
    summary = {
        "total": len(out),
        "counts": counts,
        "by_repo": by_repo,
        "deep": deep,
    }
    (DATA / "fix_verification_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_fast(start: str, end: str, queries_path: Path, keywords: list[str]) -> dict:
    """Search-based fast path: rerun each query with `linked:pr` and intersect with the
    filtered pool. Costs only 10 search calls, but only catches PR-linked closures.
    Use this as a lower-bound when GITHUB_TOKEN is missing.
    """
                                                                              
                                                                        
    issues = json.loads((DATA / "filtered_issues.json").read_text())
    full_path = DATA / "filtered_issues_full.json"
    if full_path.exists():
        full_by_url = {r["url"]: r for r in json.loads(full_path.read_text())}
        for issue in issues:
            if issue["url"] in full_by_url:
                issue.update(full_by_url[issue["url"]])

    rows = json.loads(queries_path.read_text())
    specs = [
        QuerySpec(
            name=row["name"],
            repo=row["repo"],
            labels=tuple(row["labels"]),
            extra_terms=("linked:pr",),
        )
        for row in rows
    ]

    linked_urls: set[str] = set()
    for spec in specs:
        for it in search_all(spec, start, end):
            text = f"{it.get('title', '')}\n{it.get('body', '') or ''}".lower()
            if any(k.lower() in text for k in keywords):
                linked_urls.add(it["html_url"])

    out: list[dict] = []
    counts = {"fix_verified": 0, "unverified": 0, "excluded_pull_request": 0}
    by_repo: dict[str, dict[str, int]] = {}
    for issue in issues:
        url = issue["url"]
        is_pr = bool(issue.get("pull_request"))
        if is_pr:
            verdict = "excluded_pull_request"
            flags = {"is_pull_request": True}
        elif url in linked_urls:
            verdict = "fix_verified"
            flags = {"linked_pr": True}
        else:
            verdict = "unverified"
            flags = {"linked_pr": False}
        counts[verdict] += 1
        bucket = by_repo.setdefault(issue["repo"], {"fix_verified": 0, "unverified": 0, "excluded_pull_request": 0})
        bucket[verdict] += 1
        out.append(
            {
                "url": url,
                "repo": issue["repo"],
                "number": issue["number"],
                "title": issue.get("title"),
                "verdict": verdict,
                "signals": flags,
            }
        )

    (DATA / "fix_verification.json").write_text(json.dumps(out, indent=2))
    summary = {
        "total": len(out),
        "counts": counts,
        "by_repo": by_repo,
        "mode": "fast_search_linked_pr",
    }
    (DATA / "fix_verification_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Empirical study stage 3: verify fix linkage.")
    parser.add_argument("--mode", choices=("timeline", "fast"), default="timeline",
                        help="timeline (per-issue, accurate) or fast (10 search calls, lower bound).")
    parser.add_argument("--deep", action="store_true", help="Also follow body 'Fixes #N' references.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--queries", type=Path, default=CONFIGS / "queries.json")
    parser.add_argument("--keywords", type=Path, default=CONFIGS / "keywords.json")
    args = parser.parse_args()

    if args.mode == "fast":
        keywords = json.loads(args.keywords.read_text())["correctness_keywords"]
        print(json.dumps(run_fast(args.start, args.end, args.queries, keywords), indent=2))
    else:
        print(json.dumps(run(args.deep, args.limit), indent=2))


if __name__ == "__main__":
    main()
