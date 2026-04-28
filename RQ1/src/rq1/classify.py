"""Stage 4: classify each fix-verified issue into A/B/C/D/E.

Two backends share one rubric (configs/rubric.json):

  - 'heuristic'  (default, deterministic, no API)
       Scores each category by the number of distinct signal substrings that
       appear in the issue title, body, or labels. The highest score wins.
       Ties break in the rubric's listed order (A → B → C → D); E is assigned
       only when no signal fires.

  - 'llm'        (requires ANTHROPIC_API_KEY)
       Sends each issue's title+body+labels to Claude with the rubric prompt
       and parses the returned single-letter primary category.

Inputs:
    data/fix_verification.json
    data/filtered_issues_full.json (or filtered_issues.json)
    configs/rubric.json

Outputs:
    data/classification.json
    data/classification_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable

HERE = Path(__file__).resolve().parents[2]
DATA = HERE / "data"
CONFIGS = HERE / "configs"

PRIMARY_ORDER = ("A", "B", "C", "D", "E")


def _text(issue: dict) -> str:
    body = issue.get("body") or ""
    labels = " ".join(issue.get("labels") or [])
    return f"{issue.get('title', '')}\n{body}\n{labels}".lower()


def _score_heuristic(text: str, rubric: dict) -> dict[str, int]:
    scores: dict[str, int] = {k: 0 for k in PRIMARY_ORDER}
    for letter, spec in rubric["categories"].items():
        if letter == "E":
            continue
        signals = spec.get("signals", [])
        scores[letter] = sum(1 for s in signals if s.lower() in text)
    return scores


def heuristic_label(issue: dict, rubric: dict) -> tuple[str, dict[str, int]]:
    scores = _score_heuristic(_text(issue), rubric)
    best = max(scores[c] for c in ("A", "B", "C", "D"))
    if best == 0:
        return "E", scores
    for letter in ("A", "B", "C", "D"):
        if scores[letter] == best:
            return letter, scores
    return "E", scores


def llm_label(issue: dict, rubric: dict) -> tuple[str, dict | None]:
    """Use Claude to classify. Skipped when ANTHROPIC_API_KEY is missing."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set; cannot use LLM backend")

    try:
        import anthropic  # type: ignore
    except ImportError as exc:
        raise RuntimeError("anthropic SDK not installed; pip install anthropic") from exc

    client = anthropic.Anthropic(api_key=api_key)
    prompt = _build_prompt(issue, rubric)
    msg = client.messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(part.text for part in msg.content if hasattr(part, "text")).strip()
    m = re.search(r"\b([ABCDE])\b", text)
    if not m:
        return "E", {"raw": text}
    return m.group(1), {"raw": text}


def _build_prompt(issue: dict, rubric: dict) -> str:
    cats = "\n".join(
        f"{letter}. {spec['name']}: {spec['description']}"
        for letter, spec in rubric["categories"].items()
    )
    body = (issue.get("body") or "")[:4000]
    labels = ", ".join(issue.get("labels") or [])
    return (
        "You are classifying a closed GitHub issue from PyTorch or TensorFlow into "
        "exactly one primary category. Read the rubric, then output ONLY the single "
        "letter A/B/C/D/E.\n\n"
        f"Rubric:\n{cats}\n\n"
        f"Issue URL: {issue.get('url')}\n"
        f"Title: {issue.get('title')}\n"
        f"Labels: {labels}\n"
        f"Body:\n{body}\n\n"
        "Answer with one letter."
    )


def run(backend: str, limit: int | None) -> dict:
    rubric = json.loads((CONFIGS / "rubric.json").read_text())

    fix_path = DATA / "fix_verification.json"
    if not fix_path.exists():
        raise FileNotFoundError("Run verify_fix first.")
    fix_records = {r["url"]: r for r in json.loads(fix_path.read_text())}

    src_path = DATA / "filtered_issues_full.json"
    if not src_path.exists():
        src_path = DATA / "filtered_issues.json"
    issues_by_url = {r["url"]: r for r in json.loads(src_path.read_text())}

    fix_verified = [u for u, r in fix_records.items() if r["verdict"] == "fix_verified"]

    out: list[dict] = []
    counts: dict[str, int] = {k: 0 for k in PRIMARY_ORDER}
    per_repo_counts: dict[str, dict[str, int]] = {}

    skipped_unhydrated = 0
    for i, url in enumerate(sorted(fix_verified)):
        if limit is not None and i >= limit:
            break
        issue = issues_by_url.get(url)
        if issue is None:
            continue
        if not issue.get("body") and not issue.get("labels"):
            skipped_unhydrated += 1
            continue
        if backend == "heuristic":
            letter, evidence = heuristic_label(issue, rubric)
        elif backend == "llm":
            letter, evidence = llm_label(issue, rubric)
        else:
            raise ValueError(f"unknown backend: {backend}")

        out.append(
            {
                "url": url,
                "repo": issue["repo"],
                "number": issue["number"],
                "title": issue.get("title"),
                "primary": letter,
                "category_name": rubric["categories"][letter]["name"],
                "evidence": evidence,
            }
        )
        counts[letter] += 1
        bucket = per_repo_counts.setdefault(issue["repo"], {k: 0 for k in PRIMARY_ORDER})
        bucket[letter] += 1

    (DATA / "classification.json").write_text(json.dumps(out, indent=2))

    state_count = sum(counts[k] for k in ("A", "B", "C", "D"))
    summary = {
        "backend": backend,
        "total_fix_verified": len(fix_verified),
        "classified": len(out),
        "skipped_unhydrated": skipped_unhydrated,
        "counts": counts,
        "state_related": state_count,
        "non_state": counts["E"],
        "per_repo": per_repo_counts,
    }
    (DATA / "classification_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ1 stage 4: classify into A/B/C/D/E.")
    parser.add_argument("--backend", choices=("heuristic", "llm"), default="heuristic")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.backend, args.limit), indent=2))


if __name__ == "__main__":
    main()
