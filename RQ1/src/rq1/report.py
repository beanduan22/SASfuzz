"""Stage 5: produce paper-aligned tables (Table 1 + Table 2) and a Markdown report.

Reads:
    data/collect_summary.json
    data/fix_verification_summary.json
    data/classification_summary.json   (optional)

Writes:
    reports/RQ1_report.md
    reports/RQ1_numbers.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
DATA = HERE / "data"
REPORTS = HERE / "reports"

# Numbers reported in the paper (ground-truth target the pipeline aims to confirm).
PAPER_TARGETS = {
    "raw_hits": 6011,
    "unique_hits": 5879,
    "filtered_hits": 1122,
    "fix_verified_total": 329,
    "unverified_total": 793,
    "fix_verified_pytorch": 194,
    "fix_verified_tensorflow": 135,
    "state_related_total": 206,
    "state_related_pytorch": 125,
    "state_related_tensorflow": 81,
    "non_state_total": 105,
    "udup_total": 18,
    "categories": {
        "A": {"total": 55, "pytorch": 41, "tensorflow": 14},
        "B": {"total": 74, "pytorch": 31, "tensorflow": 43},
        "C": {"total": 62, "pytorch": 43, "tensorflow": 19},
        "D": {"total": 15, "pytorch": 10, "tensorflow": 5},
    },
}


def _load(name: str) -> dict | None:
    path = DATA / name
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _diff(actual: int | None, target: int) -> str:
    if actual is None:
        return "n/a"
    return "match" if actual == target else f"Δ{actual - target:+d}"


def render_markdown(collect: dict, verify: dict | None, classify: dict | None) -> str:
    lines: list[str] = []
    lines.append("# RQ1 — Empirical Study: Live Numbers vs Paper Targets\n")
    lines.append("All counts are recomputed from `RQ1/data/`. The 'Δ' column shows the gap "
                 "vs the paper's reported number; `match` means they agree.\n")

    # Collection table
    lines.append("## Collection (raw → unique → filtered)\n")
    lines.append("| Metric | Live | Paper | Δ |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| Raw hits | {collect['raw_hits']} | {PAPER_TARGETS['raw_hits']} | {_diff(collect['raw_hits'], PAPER_TARGETS['raw_hits'])} |")
    lines.append(f"| Unique hits | {collect['unique_hits']} | {PAPER_TARGETS['unique_hits']} | {_diff(collect['unique_hits'], PAPER_TARGETS['unique_hits'])} |")
    lines.append(f"| Filtered (correctness keywords) | {collect['filtered_hits']} | {PAPER_TARGETS['filtered_hits']} | {_diff(collect['filtered_hits'], PAPER_TARGETS['filtered_hits'])} |")
    lines.append(f"| Filtered — PyTorch | {collect['by_repo']['pytorch/pytorch']} | — | — |")
    lines.append(f"| Filtered — TensorFlow | {collect['by_repo']['tensorflow/tensorflow']} | — | — |\n")

    # Per-query table
    lines.append("### Per-query counts (raw search)\n")
    lines.append("| Query | Repo | Labels | Count |")
    lines.append("|---|---|---|---:|")
    for q in collect["per_query"]:
        labels = " + ".join(q["labels"])
        lines.append(f"| {q['name']} | {q['repo']} | {labels} | {q['count']} |")
    lines.append("")

    # Fix-verification table
    if verify is None:
        lines.append("## Fix-verification\n")
        lines.append("> Stage skipped (`data/fix_verification_summary.json` missing). "
                     "Run `python -m rq1.verify_fix` with `GITHUB_TOKEN` exported.\n")
    else:
        c = verify["counts"]
        by_repo = verify["by_repo"]
        pt = by_repo.get("pytorch/pytorch", {})
        tf = by_repo.get("tensorflow/tensorflow", {})
        lines.append("## Fix-verification (Table 1)\n")
        lines.append("| Repo | Fix-verified | Unverified | Excluded (PR) | Total |")
        lines.append("|---|---:|---:|---:|---:|")
        lines.append(
            f"| pytorch/pytorch | {pt.get('fix_verified', 0)} | {pt.get('unverified', 0)} | {pt.get('excluded_pull_request', 0)} | {sum(pt.values())} |"
        )
        lines.append(
            f"| tensorflow/tensorflow | {tf.get('fix_verified', 0)} | {tf.get('unverified', 0)} | {tf.get('excluded_pull_request', 0)} | {sum(tf.values())} |"
        )
        lines.append(
            f"| **Total** | **{c['fix_verified']}** | **{c['unverified']}** | **{c.get('excluded_pull_request', 0)}** | **{verify['total']}** |\n"
        )
        lines.append("")
        lines.append("| Comparison | Live | Paper | Δ |")
        lines.append("|---|---:|---:|---:|")
        lines.append(f"| Fix-verified total | {c['fix_verified']} | {PAPER_TARGETS['fix_verified_total']} | {_diff(c['fix_verified'], PAPER_TARGETS['fix_verified_total'])} |")
        lines.append(f"| Unverified total | {c['unverified']} | {PAPER_TARGETS['unverified_total']} | {_diff(c['unverified'], PAPER_TARGETS['unverified_total'])} |")
        lines.append(f"| Fix-verified PyTorch | {pt.get('fix_verified', 0)} | {PAPER_TARGETS['fix_verified_pytorch']} | {_diff(pt.get('fix_verified', 0), PAPER_TARGETS['fix_verified_pytorch'])} |")
        lines.append(f"| Fix-verified TensorFlow | {tf.get('fix_verified', 0)} | {PAPER_TARGETS['fix_verified_tensorflow']} | {_diff(tf.get('fix_verified', 0), PAPER_TARGETS['fix_verified_tensorflow'])} |\n")

    # Classification table
    if classify is None:
        lines.append("## Classification (Table 2)\n")
        lines.append("> Stage skipped (`data/classification_summary.json` missing). "
                     "Run `python -m rq1.classify --backend heuristic` (offline) or "
                     "`--backend llm` (with `ANTHROPIC_API_KEY`).\n")
    else:
        cnt = classify["counts"]
        per_repo = classify["per_repo"]
        pt = per_repo.get("pytorch/pytorch", {k: 0 for k in "ABCDE"})
        tf = per_repo.get("tensorflow/tensorflow", {k: 0 for k in "ABCDE"})

        lines.append(f"## Classification — backend = `{classify['backend']}` (Table 2)\n")
        if classify.get("skipped_unhydrated"):
            lines.append(
                f"_Note: {classify['skipped_unhydrated']} of {classify['total_fix_verified']} fix-verified "
                "issues were skipped because their body/labels are not yet hydrated. "
                "Run `rq1.hydrate --only-fix-verified` (with `GITHUB_TOKEN` for speed) to complete classification._\n"
            )
        lines.append("| Primary dimension | PyTorch | TF | Total |")
        lines.append("|---|---:|---:|---:|")
        for letter, name in (
            ("A", "gradient_tracking"),
            ("B", "execution_mode"),
            ("C", "distribution_strategy"),
            ("D", "other_state"),
            ("E", "non_state"),
        ):
            lines.append(f"| {letter}. {name} | {pt.get(letter, 0)} | {tf.get(letter, 0)} | {cnt[letter]} |")
        state_pt = sum(pt.get(k, 0) for k in "ABCD")
        state_tf = sum(tf.get(k, 0) for k in "ABCD")
        lines.append(f"| **State-related (A–D)** | **{state_pt}** | **{state_tf}** | **{classify['state_related']}** |")
        lines.append(f"| **Non-state (E)** | **{pt.get('E', 0)}** | **{tf.get('E', 0)}** | **{classify['non_state']}** |\n")

        lines.append("")
        lines.append("| Comparison | Live | Paper | Δ |")
        lines.append("|---|---:|---:|---:|")
        lines.append(f"| State-related total | {classify['state_related']} | {PAPER_TARGETS['state_related_total']} | {_diff(classify['state_related'], PAPER_TARGETS['state_related_total'])} |")
        for letter, target in PAPER_TARGETS["categories"].items():
            lines.append(f"| {letter} total | {cnt[letter]} | {target['total']} | {_diff(cnt[letter], target['total'])} |")
            lines.append(f"| {letter} PyTorch | {pt.get(letter, 0)} | {target['pytorch']} | {_diff(pt.get(letter, 0), target['pytorch'])} |")
            lines.append(f"| {letter} TensorFlow | {tf.get(letter, 0)} | {target['tensorflow']} | {_diff(tf.get(letter, 0), target['tensorflow'])} |")

    lines.append("\n---\n")
    lines.append("See `reports/DELTA_ANALYSIS.md` for the explanation of where live numbers diverge from the paper "
                 "and what additional commands close the remaining gaps.\n")
    lines.append("Reproducibility: `python -m rq1.collect` then "
                 "`python -m rq1.hydrate` (or `--only-fix-verified`) then "
                 "`python -m rq1.verify_fix` (or `--mode fast`) then "
                 "`python -m rq1.classify --backend heuristic` then "
                 "`python -m rq1.report`.")
    return "\n".join(lines) + "\n"


def run() -> dict:
    collect = _load("collect_summary.json")
    if collect is None:
        raise FileNotFoundError("Run `python -m rq1.collect` first.")
    verify = _load("fix_verification_summary.json")
    classify = _load("classification_summary.json")

    REPORTS.mkdir(parents=True, exist_ok=True)
    md = render_markdown(collect, verify, classify)
    (REPORTS / "RQ1_report.md").write_text(md)

    final = {
        "collect": collect,
        "verify": verify,
        "classify": classify,
        "paper_targets": PAPER_TARGETS,
    }
    (REPORTS / "RQ1_numbers.json").write_text(json.dumps(final, indent=2))
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ1 stage 5: render report.")
    parser.parse_args()
    summary = run()
    print(json.dumps({k: (v if k == "paper_targets" else (v or "missing")) for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
