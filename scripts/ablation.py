"""
RQ4: Ablation study runner.

Runs SASFuzz with each ablation variant on both frameworks and reports
the metrics table matching paper Table 5.

Usage:
    python scripts/ablation.py --models 50 --budget 86400
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).parent.parent

                                                                           
ABLATIONS = ["none", "no_skeleton", "no_scaffold", "no_selection"]
ABLATION_LABELS = {
    "none":         "SASFuzz",
    "no_skeleton":  "w/o Skeleton",
    "no_scaffold":  "w/o Scaffold",
    "no_selection": "w/o State Rel.",
}

                                                                   
LLM_BACKENDS = ["gpt5", "qwen"]
LLM_LABELS   = {
    "gpt5":        "GPT-5",
    "qwen":        "Qwen3.6-27B",
}

def run_variant(out_dir, framework, ablation, llm_backend, models, budget, api_set_size):
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    if framework == "pytorch":
        cmd = [
            sys.executable, str(HERE / "run_pytorch.py"),
            "--mode", "full", "--models", str(models),
            "--budget", str(budget),
            "--api-set-size", str(api_set_size),
            "--llm-backend", llm_backend,
            "--ablation", ablation,
            "--output-dir", str(out_dir),
        ]
    else:
        cmd = [
            sys.executable, str(HERE / "run_tensorflow.py"),
            "--models", str(models),
            "--budget", str(budget),
            "--api-set-size", str(api_set_size),
            "--llm-backend", llm_backend,
            "--ablation", ablation,
            "--out", str(out_dir),
        ]
    with log_path.open("w") as lf:
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(HERE.parent))
    return p, log_path

def read_metrics(out_dir):
    cov_file = out_dir / "coverage.json"
    if not cov_file.exists():
        return {}
    d = json.loads(cov_file.read_text())
    n_exec  = d.get("n_executed", 0)
    n_total = max(d.get("n_total_apis", 1), 1)
    totals  = d.get("totals", {})
    n_models = d.get("n_models", 0)
    failed = totals.get("failed_synth", 0) + totals.get("invalid_models", 0)
    succ_rate = 100 * (n_models - failed) / max(n_models, 1)
    return {
        "succ_pct": succ_rate,
        "n_api": n_exec,
        "code_pct": 100 * n_exec / n_total,
        "n_cases": totals.get("bugs", 0),
    }

def main():
    ap = argparse.ArgumentParser(description="RQ4: ablation + LLM sensitivity")
    ap.add_argument("--models",       type=int, default=1000)
    ap.add_argument("--budget",       type=int, default=86400)
    ap.add_argument("--api-set-size", type=int, default=30)
    ap.add_argument("--out",          default=str(HERE/"results"/"rq4"))
    ap.add_argument("--experiment",   choices=["ablation","llm","both"], default="both")
    ap.add_argument("--framework",    choices=["pytorch","tensorflow","both"], default="pytorch")
    ap.add_argument("--llm-backend",  default="gpt5",
                    choices=["gpt5","qwen"],
                    help="Backend for component ablation (paper default: gpt5)")
    args = ap.parse_args()

    out_root = Path(args.out)
    frameworks = ["pytorch","tensorflow"] if args.framework == "both" else [args.framework]

    if args.experiment in ("ablation","both"):
        print("=== RQ4 Part 1: Component Ablation ===")
        ablation_procs = {}
        for fw in frameworks:
            for abl in ABLATIONS:
                out_dir = out_root / "ablation" / fw / abl
                p, log = run_variant(out_dir, fw, abl, args.llm_backend,
                                     args.models, args.budget, args.api_set_size)
                ablation_procs[(fw, abl)] = (p, out_dir)
                print(f"  Started {fw}/{abl}  PID={p.pid}")

        print("  Waiting for ablation runs to finish...")
        while any(p.poll() is None for p, _ in ablation_procs.values()):
            time.sleep(30)

        print("\nAblation Results (matches paper Table 5):")
        for fw in frameworks:
            print(f"\n  {fw.upper()}")
            hdr = f"  {'Setting':<22} {'%Succ':>7} {'#API':>6} {'%Code':>7} {'#Cases':>8}"
            print(hdr)
            print("  " + "-"*(len(hdr)-2))
            for abl in ABLATIONS:
                _, out_dir = ablation_procs[(fw, abl)]
                m = read_metrics(out_dir)
                label = ABLATION_LABELS[abl]
                if m:
                    print(f"  {label:<22} {m['succ_pct']:>6.2f}% {m['n_api']:>6} {m['code_pct']:>6.2f}% {m['n_cases']:>8}")
                else:
                    print(f"  {label:<22} {'N/A':>7}")

    if args.experiment in ("llm","both"):
        print("\n=== RQ4 Part 2: LLM Sensitivity ===")
        llm_procs = {}
        for fw in frameworks:
            for backend in LLM_BACKENDS:
                out_dir = out_root / "llm" / fw / backend
                p, log = run_variant(out_dir, fw, "none", backend,
                                     args.models, args.budget, args.api_set_size)
                llm_procs[(fw, backend)] = (p, out_dir)
                print(f"  Started {fw}/{backend}  PID={p.pid}")

        print("  Waiting for LLM sensitivity runs to finish...")
        while any(p.poll() is None for p, _ in llm_procs.values()):
            time.sleep(30)

        print("\nLLM Sensitivity Results (matches paper Table 6):")
        for fw in frameworks:
            print(f"\n  {fw.upper()}")
            hdr = f"  {'LLM':<16} {'%Succ':>7} {'#API':>6} {'%Code':>7} {'#Cases':>8}"
            print(hdr)
            print("  " + "-"*(len(hdr)-2))
            for backend in LLM_BACKENDS:
                _, out_dir = llm_procs[(fw, backend)]
                m = read_metrics(out_dir)
                label = LLM_LABELS[backend]
                if m:
                    print(f"  {label:<16} {m['succ_pct']:>6.2f}% {m['n_api']:>6} {m['code_pct']:>6.2f}% {m['n_cases']:>8}")
                else:
                    print(f"  {label:<16} {'N/A':>7}")

if __name__ == "__main__":
    main()
