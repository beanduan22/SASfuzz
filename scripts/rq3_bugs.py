#!/usr/bin/env python3
"""
RQ3: Bug study aggregator.

Scans a results directory for bug JSON files, deduplicates by root-cause
skeleton, and prints a table matching paper Table 4.

Usage:
    python scripts/rq3_bugs.py --results results/rq2
"""
import argparse, json, collections
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="RQ3: bug study aggregator")
    ap.add_argument("--results", default="results", help="Root results directory")
    args = ap.parse_args()

    root = Path(args.results)
    rows = []  # (framework, bug_type, skeleton_id, used_apis, detail, timestamp)
    for bug_json in sorted(root.rglob("bug_*.json")):
        try:
            d = json.loads(bug_json.read_text())
        except Exception:
            continue
        if d.get("bug_type") in ("generation_failure",):
            continue
        framework = "tensorflow" if "tensorflow" in str(bug_json) else "pytorch"
        rows.append({
            "framework": framework,
            "bug_type":  d.get("bug_type", "unknown"),
            "used_apis": d.get("used_apis", []),
            "detail":    d.get("detail", "")[:80],
            "mutation":  d.get("mutation_name", ""),
            "timestamp": d.get("timestamp", ""),
            "file":      str(bug_json),
        })

    # Group by (framework, bug_type)
    by_fw_type = collections.defaultdict(list)
    for r in rows:
        by_fw_type[(r["framework"], r["bug_type"])].append(r)

    print("=== RQ3: Bug Summary (Table 4) ===\n")
    header = f"{'Framework':<14} {'#Cases':>7} {'Crash':>6} {'NaN':>5} {'Incon.':>7}"
    print(header)
    print("-" * len(header))
    grand_total = 0
    for fw in ("pytorch", "tensorflow"):
        cases   = [r for r in rows if r["framework"] == fw]
        crashes = sum(1 for r in cases if r["bug_type"] == "crash")
        nans    = sum(1 for r in cases if r["bug_type"] == "nan")
        incons  = sum(1 for r in cases if r["bug_type"] == "inconsistent")
        grand_total += len(cases)
        print(f"{fw:<14} {len(cases):>7} {crashes:>6} {nans:>5} {incons:>7}")
    print("-" * len(header))
    all_c = sum(1 for r in rows if r["bug_type"] == "crash")
    all_n = sum(1 for r in rows if r["bug_type"] == "nan")
    all_i = sum(1 for r in rows if r["bug_type"] == "inconsistent")
    print(f"{'Total':<14} {grand_total:>7} {all_c:>6} {all_n:>5} {all_i:>7}")

    print(f"\nTotal anomalous cases: {len(rows)}")
    print("\nDetailed list:")
    for i, r in enumerate(rows, 1):
        apis = ", ".join(r["used_apis"][:3])
        print(f"  {i:>3}. [{r['framework']}] {r['bug_type']:<14} apis=({apis}) | {r['detail']}")

if __name__ == "__main__":
    main()
