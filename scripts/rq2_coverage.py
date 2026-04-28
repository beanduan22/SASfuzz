"""
RQ2: API coverage and code coverage comparison runner.

Runs SASFuzz for N models and reports coverage metrics that can be compared
against baselines (Muffin, COMET, ModelMeta) run separately.

Usage:
    python scripts/rq2_coverage.py --framework pytorch --models 100 --budget 60
    python scripts/rq2_coverage.py --framework tensorflow --models 100 --budget 60
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).parent.parent

def main():
    ap = argparse.ArgumentParser(description="RQ2: coverage runner")
    ap.add_argument("--framework", choices=["pytorch","tensorflow","both"], default="both")
    ap.add_argument("--models",    type=int, default=100)
    ap.add_argument("--budget",    type=int, default=60)
    ap.add_argument("--api-set-size", type=int, default=30)
    ap.add_argument("--llm-backend",  default="ollama")
    ap.add_argument("--out",          default=str(HERE/"results"/"rq2"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    procs = []
    if args.framework in ("pytorch", "both"):
        pt_out = out / "pytorch"
        pt_out.mkdir(exist_ok=True)
        cmd = [
            sys.executable, "-m", "smolfuzz.main",
            "--mode", "full", "--models", str(args.models),
            "--budget", str(args.budget),
            "--api-set-size", str(args.api_set_size),
            "--llm-backend", args.llm_backend,
            "--output-dir", str(pt_out),
        ]
        log = (out / "pytorch.log").open("w")
        procs.append(("pytorch", subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(HERE.parent))))
        print(f"PyTorch fuzzer started (PID {procs[-1][1].pid}) → {pt_out}")

    if args.framework in ("tensorflow", "both"):
        tf_out = out / "tensorflow"
        tf_out.mkdir(exist_ok=True)
        cmd = [
            sys.executable, str(HERE/"run_tf.py"),
            "--models", str(args.models),
            "--budget", str(args.budget),
            "--api-set-size", str(args.api_set_size),
            "--llm-backend", args.llm_backend,
            "--out", str(tf_out),
        ]
        log = (out / "tensorflow.log").open("w")
        procs.append(("tensorflow", subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(HERE.parent))))
        print(f"TensorFlow fuzzer started (PID {procs[-1][1].pid}) → {tf_out}")

    start = time.time()
    while any(p.poll() is None for _, p in procs):
        time.sleep(30)
        for name, p in procs:
            rc = p.poll()
            elapsed = (time.time() - start) / 60
            print(f"  [{elapsed:.1f}min] {name}: {'DONE' if rc is not None else 'running'}")

    print("\n=== RQ2 Coverage Report ===")
    for name in ("pytorch", "tensorflow"):
        cov_file = out / name / "coverage.json"
        if not cov_file.exists():
            continue
        data = json.loads(cov_file.read_text())
        n_exec  = data.get("n_executed", 0)
        n_total = data.get("n_total_apis", 1)
        totals  = data.get("totals", {})
        print(f"\n{name.upper()}:")
        print(f"  APIs executed : {n_exec} / {n_total} ({100*n_exec/n_total:.1f}%)")
        print(f"  Models        : {data.get('n_models', 0)}")
        print(f"  Bugs found    : {totals.get('bugs', 0)}")
        print(f"  Results       → {out / name}")

if __name__ == "__main__":
    main()
