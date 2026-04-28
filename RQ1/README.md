# RQ1 — Empirical Study Pipeline

Code that reproduces RQ1 of the paper: *Are state-related issues prevalent in
correctness-oriented issue reports of modern DL libraries, and if so, which
state dimensions do they concentrate on?*

Pipeline:

```
collect.py  →  hydrate.py  →  verify_fix.py  →  classify.py  →  report.py
   |               |               |                 |              |
   ▼               ▼               ▼                 ▼              ▼
raw_search    filtered_       fix_              classification    reports/
filtered_     issues_full    verification                          RQ1_report.md
issues
```

## Quick start

```bash
cd RQ1

# Stage 1 — by default replays the v2 search snapshot at
#           ../results/github_issue_counts_v2.json (1122 filtered issues).
PYTHONPATH=src python3 -m rq1.collect

# Stage 2 — fetch full title/body/labels for the 1122 issues.
#           Requires GITHUB_TOKEN. Skippable if a hydrate cache already exists.
GITHUB_TOKEN=ghp_... PYTHONPATH=src python3 -m rq1.hydrate

# Stage 3 — determine fix-verified vs unverified via the GitHub timeline API.
GITHUB_TOKEN=ghp_... PYTHONPATH=src python3 -m rq1.verify_fix --deep

# Stage 4 — classify fix-verified issues into A/B/C/D/E.
PYTHONPATH=src python3 -m rq1.classify --backend heuristic
# OR for the LLM rater used in the paper:
ANTHROPIC_API_KEY=sk-... PYTHONPATH=src python3 -m rq1.classify --backend llm

# Stage 5 — render reports/RQ1_report.md with a side-by-side vs the paper.
PYTHONPATH=src python3 -m rq1.report
```

`make confirm` runs the offline portion (collect + report) and prints the
report.

## Stage I/O contract

| Stage | Input | Output |
|---|---|---|
| collect      | `configs/queries.json`, `configs/keywords.json` | `data/raw_search.json`, `data/filtered_issues.json`, `data/collect_summary.json` |
| hydrate      | `data/filtered_issues.json` | `data/filtered_issues_full.json`, `data/issue_cache/<repo>/<n>.json` |
| verify_fix   | `data/filtered_issues_full.json` | `data/fix_verification.json`, `data/timeline_cache/<repo>/<n>.json` |
| classify     | `data/fix_verification.json` (+ rubric) | `data/classification.json`, `data/classification_summary.json` |
| report       | all of the above | `reports/RQ1_report.md`, `reports/RQ1_numbers.json` |

## What the pipeline confirms

The paper reports the following numbers; the pipeline recomputes each and the
report renders a side-by-side table:

| Metric | Paper |
|---|---:|
| Raw hits | 6,011 |
| Unique hits | 5,879 |
| Filtered hits | 1,122 |
| Fix-verified (PT) | 194 |
| Fix-verified (TF) | 135 |
| State-related (A–D) | 206 |
| A. gradient tracking | 55 |
| B. execution mode | 74 |
| C. distribution strategy | 62 |
| D. other state | 15 |
| E. non-state | 105 |

The collect stage is fully reproducible offline against the cached search
snapshot. Stages 2–4 require GitHub API access (and, for the LLM backend, an
Anthropic API key).
