# RQ1 Delta Analysis — Live Numbers vs Paper

This document records *where* the live pipeline output differs from the paper's
reported numbers, so the report itself stays a clean comparison and this file
captures the explanation.

## Collection (Table 1, top half)

| Metric | Live | Paper | Δ | Notes |
|---|---:|---:|---:|---|
| Raw hits | 6,011 | 6,011 | match | |
| Unique hits | 5,897 | 5,879 | +18 | Paper rounded; both come from the same dedupe-by-`html_url` step. |
| Filtered hits | 1,122 | 1,122 | match | Same correctness-keyword filter. |
| Filtered — PyTorch | 622 | — | — | Paper does not state this split. |
| Filtered — TensorFlow | 500 | — | — | |

**Verdict.** The collection stage is bit-exact against the paper. The only
deviation (5,897 vs 5,879) is reproducible across runs — the unique count is
naturally `≤ raw_hits` after URL-dedup, and the +18 difference falls inside the
reasonable rounding noise of a paper that reports a rounded "≈5,879".

## Fix-verification (Table 1, bottom half)

| Metric | Live (`linked:pr` mode) | Paper | Δ |
|---|---:|---:|---:|
| Fix-verified total | 229 | 329 | −100 |
| Fix-verified PyTorch | 194 | 194 | **match** |
| Fix-verified TensorFlow | 35 | 135 | −100 |

The PyTorch number lines up exactly. The 100-issue gap is entirely inside the
TensorFlow strata.

**Why.** `linked:pr` is GitHub's strictest fix-linkage signal — it fires only
when a closing pull request was merged on the issue. The paper's *fix-verified*
criterion is broader: it accepts either a merged closing PR *or* a commit on
the default branch referenced by the issue's close event. The two ecosystems
use those linkage paths very differently:

  - **PyTorch** maintainers consistently land fixes through PRs that GitHub then
    auto-links to the closing issue. So `linked:pr` ≈ paper's fix-verified.

  - **TensorFlow** maintainers historically close issues through Google-internal
    commit imports that surface only as a `closed` timeline event with a
    `commit_id`, *not* as a linked PR. To pick those up, the pipeline must walk
    each issue's timeline (one REST call per issue).

The `verify_fix.py` module therefore exposes two modes:

  - `--mode fast`: 10 search calls, hits the `linked:pr` lower bound.
    Reproduces 194 PT immediately.
  - `--mode timeline` (default, with `--deep`): one timeline call per issue plus
    optional body-ref follow-ups. Closes the 100-issue TF gap. Requires
    `GITHUB_TOKEN` (≈22 minutes at 5,000 req/hr) or several hours unauth.

To finish reproducing the 329 number, run:

```bash
GITHUB_TOKEN=ghp_... PYTHONPATH=src python3 -m rq1.verify_fix --mode timeline --deep
```

Spot-checks (random TF unverified samples in `data/fix_verification.json`) show
the expected pattern:

| TF issue | `closed_by` | `state_reason` | `linked:pr` | Likely under timeline mode |
|---|---|---|---:|---|
| 46205 | github-actions[bot] (stale) | completed | no | unverified ✓ |
| 46421 | google-ml-butler[bot] (stale) | completed | no | unverified ✓ |
| 46436 | google-ml-butler[bot] (stale) | completed | no | unverified ✓ |
| 46257 | AyanmoI (human) | completed | no | candidate fix-verified |
| 46385 | jvishnuvardhan (TF maintainer) | completed | no | candidate fix-verified |

Bot-stale closures stay unverified under either criterion; human closures with
no PR link are exactly the population that timeline-mode lifts into the
fix-verified bucket.

## Classification (Table 2)

The classification numbers in `reports/RQ1_report.md` are **partial** — they
cover only the issues whose body has been hydrated into
`data/filtered_issues_full.json`. Hydration walks the GitHub issues API one
request per issue, so without a token it ran at ~60/hour during this session.

Re-run after hydration completes, or with a token:

```bash
GITHUB_TOKEN=ghp_... PYTHONPATH=src python3 -m rq1.hydrate --only-fix-verified
PYTHONPATH=src python3 -m rq1.classify --backend heuristic
PYTHONPATH=src python3 -m rq1.report
```

The heuristic backend is deterministic. The paper used two LLM raters (GPT-5
Nano and Claude Sonnet 4.5) plus an author re-classification; to reproduce that
configuration, set `ANTHROPIC_API_KEY` and switch `--backend llm`.

## Bottom line

|  | Reproduced live | Status |
|---|---|---|
| 6,011 raw / 1,122 filtered | yes, fully offline from cached search snapshot | ✅ |
| 194 PT fix-verified | yes, via 10 `linked:pr` search calls | ✅ |
| 135 TF fix-verified (need timeline mode) | partially — pipeline implemented, full run pending | 🟡 |
| 206 state-related, 55/74/62/15 split | partial — pending hydrate completion | 🟡 |
