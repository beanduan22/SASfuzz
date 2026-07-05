from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
DATA = HERE / "data"
REPORTS = HERE / "reports"

                                              
PAPER_COLLECTION = {
    "raw_hits": 6011,
    "unique_hits": 5897,
    "filtered_hits": 1122,
    "fix_verified_total": 329,
    "unverified_total": 793,
}

PAPER_FIX_VERIFIED = {
    "pytorch": {"issues": 194, "state": 125, "non_state": 58, "udup": 11, "state_pct": 64.4},
    "tensorflow": {"issues": 135, "state": 81, "non_state": 47, "udup": 7, "state_pct": 60.0},
    "total": {"issues": 329, "state": 206, "non_state": 105, "udup": 18, "state_pct": 62.6},
}

PAPER_DIMENSIONS = {
    "A": {
        "name": "Gradient tracking",
        "pytorch": {"count": 41, "pct": 32.8},
        "tensorflow": {"count": 14, "pct": 17.3},
        "total": {"count": 55, "pct": 26.7},
    },
    "B": {
        "name": "Execution mode",
        "pytorch": {"count": 31, "pct": 24.8},
        "tensorflow": {"count": 43, "pct": 53.1},
        "total": {"count": 74, "pct": 35.9},
    },
    "C": {
        "name": "Distribution strategy",
        "pytorch": {"count": 43, "pct": 34.4},
        "tensorflow": {"count": 19, "pct": 23.5},
        "total": {"count": 62, "pct": 30.1},
    },
    "D": {
        "name": "Other state",
        "pytorch": {"count": 10, "pct": 8.0},
        "tensorflow": {"count": 5, "pct": 6.2},
        "total": {"count": 15, "pct": 7.3},
    },
    "A_C_total": {
        "name": "A--C total",
        "pytorch": {"count": 115, "pct": 92.0},
        "tensorflow": {"count": 76, "pct": 93.8},
        "total": {"count": 191, "pct": 92.7},
    },
}

PAPER_NUMBERS = {
    "collection": PAPER_COLLECTION,
    "fix_verified_table": PAPER_FIX_VERIFIED,
    "dimension_table": PAPER_DIMENSIONS,
}


def _pct(value: float) -> str:
    return f"{value:.1f}%"


def render_markdown() -> str:
    lines: list[str] = []
    lines.append("# Empirical Study Numbers\n")
    lines.append(
        "This report is pinned to the paper audit dataset. "
        "The live GitHub collection scripts remain available, but this artifact reports the exact "
        "numbers used in the manuscript.\n"
    )

    lines.append("## Collection\n")
    lines.append("| Metric | Count |")
    lines.append("|---|---:|")
    lines.append(f"| Raw issues returned by repository-label queries | {PAPER_COLLECTION["raw_hits"]:,} |")
    lines.append(f"| Unique issues after URL deduplication | {PAPER_COLLECTION["unique_hits"]:,} |")
    lines.append(f"| Correctness-relevant reports after keyword filtering | {PAPER_COLLECTION["filtered_hits"]:,} |")
    lines.append(f"| Fix-verified issues | {PAPER_COLLECTION["fix_verified_total"]:,} |")
    lines.append(f"| Unverified issues | {PAPER_COLLECTION["unverified_total"]:,} |\n")

    lines.append("## Table 1: Counts for the 329 Fix-Verified Issues\n")
    lines.append("| Stratum | Framework | # Issues | State | Non (E) | U/DUP | %State |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for label, fw in (("Fix-verified", "pytorch"), ("Fix-verified", "tensorflow")):
        row = PAPER_FIX_VERIFIED[fw]
        fw_name = "PyTorch" if fw == "pytorch" else "TensorFlow"
        lines.append(
            f"| {label} | {fw_name} | {row["issues"]} | {row["state"]} | "
            f"{row["non_state"]} | {row["udup"]} | {_pct(row["state_pct"])} |"
        )
    row = PAPER_FIX_VERIFIED["total"]
    lines.append(
        f"|  | Total | {row["issues"]} | {row["state"]} | {row["non_state"]} | "
        f"{row["udup"]} | {_pct(row["state_pct"])} |\n"
    )

    lines.append("## Table 2: Distribution over the 206 State-Related Issues\n")
    lines.append("| Primary dimension | PyTorch # | PyTorch % | TensorFlow # | TensorFlow % | Total # | Total % |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for key in ("A", "B", "C", "D", "A_C_total"):
        row = PAPER_DIMENSIONS[key]
        prefix = "A--C total" if key == "A_C_total" else f"{key}. {row["name"]}"
        lines.append(
            f"| {prefix} | {row["pytorch"]["count"]} | {_pct(row["pytorch"]["pct"])} | "
            f"{row["tensorflow"]["count"]} | {_pct(row["tensorflow"]["pct"])} | "
            f"{row["total"]["count"]} | {_pct(row["total"]["pct"])} |"
        )
    lines.append("")
    return "\n".join(lines)


def run() -> dict:
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "empirical_study_report.md").write_text(render_markdown())
    (REPORTS / "empirical_study_numbers.json").write_text(json.dumps(PAPER_NUMBERS, indent=2))
    return PAPER_NUMBERS


def main() -> None:
    parser = argparse.ArgumentParser(description="Empirical study stage 5: render paper empirical study tables.")
    parser.parse_args()
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
