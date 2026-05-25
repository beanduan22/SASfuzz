from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent.parent


def _run_fuzzer(framework: str, models: int, budget: int, api_set_size: int,
                llm_backend: str, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "run.log"
    if framework == "pytorch":
        cmd = [
            sys.executable, str(HERE / "run_pytorch.py"),
            "--mode", "full",
            "--models", str(models),
            "--budget", str(budget),
            "--api-set-size", str(api_set_size),
            "--llm-backend", llm_backend,
            "--output-dir", str(out),
        ]
    else:
        cmd = [
            sys.executable, str(HERE / "run_tensorflow.py"),
            "--models", str(models),
            "--budget", str(budget),
            "--api-set-size", str(api_set_size),
            "--llm-backend", llm_backend,
            "--out", str(out),
        ]
    with log_path.open("w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                cwd=str(HERE))
    print(f"  [{framework}] PID {proc.pid} → {out}")
    return proc, log_path


def _coverage_report(out: Path, framework: str) -> None:
    cov_file = out / "coverage.json"
    if not cov_file.exists():
        print(f"  [{framework}] coverage.json not found")
        return
    d = json.loads(cov_file.read_text())
    n_exec = d.get("n_executed", 0)
    n_total = max(d.get("n_total_apis", 1), 1)
    totals = d.get("totals", {})
    n_models = d.get("n_models", 0)
    failed = totals.get("failed_synth", 0) + totals.get("invalid_models", 0)
    succ = 100 * (n_models - failed) / max(n_models, 1)
    print(f"\n  [{framework.upper()}] Coverage")
    print(f"    Models synthesised : {n_models}")
    print(f"    Synthesis success  : {succ:.1f}%")
    print(f"    APIs executed      : {n_exec} / {n_total} ({100*n_exec/n_total:.1f}%)")
    print(f"    Bugs detected      : {totals.get('bugs', 0)}")


def _bug_report(out: Path, framework: str) -> None:
    rows = []
    for bug_json in sorted(out.rglob("bug_*.json")):
        try:
            d = json.loads(bug_json.read_text())
        except Exception:
            continue
        if d.get("bug_type") == "generation_failure":
            continue
        rows.append({
            "bug_type": d.get("bug_type", "unknown"),
            "used_apis": d.get("used_apis", []),
            "detail":    d.get("detail", "")[:80],
            "mutation":  d.get("mutation_name", ""),
        })

    if not rows:
        print(f"\n  [{framework.upper()}] No bugs detected")
        return

    by_type = collections.Counter(r["bug_type"] for r in rows)
    print(f"\n  [{framework.upper()}] Bug Summary")
    print(f"    Total cases : {len(rows)}")
    print(f"    Crash       : {by_type.get('crash', 0)}")
    print(f"    NaN         : {by_type.get('nan', 0)}")
    print(f"    Inconsistent: {by_type.get('inconsistent', 0)}")
    print(f"\n    Detailed list:")
    for i, r in enumerate(rows, 1):
        apis = ", ".join(r["used_apis"][:3])
        print(f"      {i:>3}. [{r['bug_type']:<14}] {apis} | {r['detail']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="SASFuzz evaluation: coverage + bugs (RQ2 & RQ3)")
    ap.add_argument("--framework", choices=["pytorch", "tensorflow", "both"], default="both")
    ap.add_argument("--models",       type=int, default=1000)
    ap.add_argument("--budget",       type=int, default=86400)
    ap.add_argument("--api-set-size", type=int, default=30)
    ap.add_argument("--llm-backend",  default="gpt5",
                    choices=["gpt5", "qwen"])
    ap.add_argument("--out",          default=str(HERE / "results" / "eval"))
    ap.add_argument("--report-only",  action="store_true",
                    help="Skip fuzzing; only report from existing results")
    args = ap.parse_args()

    out_root = Path(args.out)
    frameworks = ["pytorch", "tensorflow"] if args.framework == "both" else [args.framework]

    if not args.report_only:
        print(f"=== SASFuzz Evaluation | {args.framework} | {args.models} models ===\n")
        procs = []
        for fw in frameworks:
            out_dir = out_root / fw
            proc, log = _run_fuzzer(fw, args.models, args.budget, args.api_set_size,
                                    args.llm_backend, out_dir)
            procs.append((fw, proc, out_dir))

        start = time.time()
        while any(p.poll() is None for _, p, _ in procs):
            time.sleep(300)
            elapsed = (time.time() - start) / 3600
            for fw, p, _ in procs:
                status = "DONE" if p.poll() is not None else "running"
                print(f"  [{elapsed:.1f}h] {fw}: {status}")

    print("\n=== Results ===")
    for fw in frameworks:
        out_dir = out_root / fw
        _coverage_report(out_dir, fw)
        _bug_report(out_dir, fw)


if __name__ == "__main__":
    main()
