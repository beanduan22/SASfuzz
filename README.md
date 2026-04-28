# SASFuzz: State-Aware Fuzzing of Deep Learning Libraries via Skeleton-Guided Synthesis

SASFuzz is a state-aware fuzzer for PyTorch and TensorFlow. It decouples *state exposure* from *API composition*: a library of 12 state skeletons fixes the program structure and state-installing constructs, while a multi-roulette selector and an LLM fill the typed slots with diverse API chains. Bugs are detected by differential testing (CPU vs. GPU) and analytical–numerical gradient checking.

---

## Repository layout

```
smolfuzz/
├── main.py                  PyTorch fuzzer entry point
├── run_tf.py                TensorFlow fuzzer entry point
├── run_both.py              Run both frameworks in parallel
│
├── backends/
│   └── llm_client.py        LLM clients: Ollama, OpenAI, Anthropic, DeepSeek + factory
│
├── core/
│   ├── api_loader.py        Load and classify the runtime API pool
│   ├── executor.py          Subprocess runner: CPU/GPU execution + 5 mutation strategies
│   ├── oracle.py            Differential oracle (crash / NaN / inconsistency) + gradcheck
│   ├── prompts.py           Synthesis, repair, free-form, and no-scaffold prompts
│   ├── selector.py          Multi-roulette selector with Eq. 1 scoring
│   ├── skeletons.py         12 state skeletons (6 PyTorch + 6 TensorFlow)
│   ├── state_signals.py     σ and b tables for Eq. 1 (offline probe + bug prior)
│   └── synthesizer.py       LLM synthesis loop with repair
│
├── scripts/
│   ├── rq2_coverage.py      RQ2: launch coverage runs, print Table 3
│   ├── rq3_bugs.py          RQ3: aggregate bug JSONs, print Table 4
│   └── rq4_ablation.py      RQ4: ablation study + LLM sensitivity, print Tables 5 & 6
│
├── RQ1/                     Empirical study pipeline (Section 2)
│   └── src/rq1/
│       ├── collect.py       GitHub issue collection
│       ├── hydrate.py       Issue body hydration
│       ├── classify.py      LLM-assisted classification
│       ├── verify_fix.py    Fix-verification filter
│       └── report.py        Table 1 & 2 generation
│
├── bugs/                    Confirmed bug reproducers
│   ├── reproducers/         Minimal standalone scripts
│   └── github/new/          New bugs filed during evaluation
│
├── torch_valid_apis.txt     PyTorch runtime API pool
└── requirements.txt
```

---

## Installation

```bash
pip install torch>=2.9.0 tensorflow>=2.21.0 numpy requests

# Install only the backends you use
pip install openai      # GPT-5 / DeepSeek-API
pip install anthropic   # Claude-3.5-Sonnet
# Qwen2.5-32B and DeepSeek-V2 run via Ollama (no pip package needed)
```

Set API keys for cloud backends:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export DEEPSEEK_API_KEY=sk-...       # only for --llm-backend deepseek-api
```

---

## Quick start

### PyTorch

```bash
python -m smolfuzz.main --mode full --models 100 --budget 60 \
       --llm-backend ollama --output-dir results/pytorch
```

### TensorFlow

```bash
python smolfuzz/run_tf.py --models 100 --budget 60 \
       --llm-backend ollama --out results/tensorflow
```

### Both frameworks in parallel

```bash
python smolfuzz/run_both.py --models 100 --budget 60
```

---

## LLM backends

| Flag | Backend | Default model | Requires |
|---|---|---|---|
| `ollama` | Local Ollama | Qwen2.5-Coder-32B + DeepSeek-V2 (round-robin) | Ollama running |
| `qwen25-32b` | Local Ollama | Qwen2.5-Coder-32B | Ollama running |
| `deepseek-v2` | Local Ollama | DeepSeek-V2 | Ollama running |
| `gpt5` | OpenAI API | gpt-4o | `OPENAI_API_KEY` |
| `claude35` | Anthropic API | claude-3-5-sonnet-20241022 | `ANTHROPIC_API_KEY` |
| `deepseek-api` | DeepSeek API | deepseek-chat | `DEEPSEEK_API_KEY` |

```bash
python -m smolfuzz.main --mode full --models 100 \
       --llm-backend claude35 --output-dir results/claude35
```

---

## State skeletons

| ID | State Dimension | Fixed Scaffold |
|---|---|---|
| PT-G1 | Gradient tracking | `requires_grad` + `backward()` |
| PT-G2 | Gradient tracking | `torch.no_grad` scope |
| PT-M1 | Execution mode | `train()` / `eval()` switch |
| PT-M2 | Execution mode | `torch.jit.trace` |
| PT-D1 | Distribution strategy | `DistributedDataParallel` |
| PT-D2 | Distribution strategy | `torch.distributed` collective |
| TF-G1 | Gradient tracking | `GradientTape` + `jacobian` |
| TF-G2 | Gradient tracking | `stop_gradient` inside tape |
| TF-M1 | Execution mode | `tf.function` tracing |
| TF-M2 | Execution mode | `tf.function(jit_compile=True)` |
| TF-D1 | Distribution strategy | `MirroredStrategy` scope |
| TF-D2 | Distribution strategy | `tf.distribute` collective |

---

## Oracles

Three oracle types (Section 3.5):

- **Crash** — unexpected exception, abort, or segfault on either device
- **NaN** — asymmetric NaN/Inf between CPU and GPU outputs
- **Inconsistency** — CPU–GPU numerical difference exceeding tolerance (~1e-2)
- **Gradcheck** — analytical–numerical Jacobian mismatch via `torch.autograd.gradcheck` (PT-G1 skeleton)

---

## Reproducing paper results

### RQ1 — Empirical study (Tables 1 & 2)

```bash
cd RQ1
python -m rq1.collect    # fetch GitHub issues
python -m rq1.hydrate    # hydrate issue bodies
python -m rq1.classify   # LLM-assisted classification
python -m rq1.report     # print Tables 1 & 2
```

Requires a GitHub token (`GITHUB_TOKEN`) and at least one LLM API key.

### RQ2 — Coverage comparison (Table 3)

```bash
python scripts/rq2_coverage.py --framework both --models 100 --budget 60
```

Reports API coverage and code coverage for SASFuzz. Baselines (Muffin, COMET, ModelMeta) must be run separately with their original artifacts.

### RQ3 — Bug study (Table 4)

```bash
python scripts/rq3_bugs.py --results results/
```

Scans all `bug_*.json` files and prints a Table 4–style breakdown by framework and bug type.

### RQ4 — Ablation + LLM sensitivity (Tables 5 & 6)

```bash
# Component ablation (Table 5)
python scripts/rq4_ablation.py --experiment ablation --models 50 --budget 60

# LLM sensitivity (Table 6)
python scripts/rq4_ablation.py --experiment llm --models 50 --budget 60
```

Ablation variants:

| `--ablation` flag | Paper row |
|---|---|
| `none` | SASFuzz (full) |
| `no_skeleton` | w/o Skeleton |
| `no_scaffold` | w/o Scaffold |
| `no_selection` | w/o Selection |
| `no_feedback` | w/o Feedback |

```bash
# Single variant, manually
python -m smolfuzz.main --mode full --models 100 \
       --ablation no_selection --output-dir results/ablation/no_selection
```

---

## Output structure

```
results/<run>/
├── coverage.json          API coverage + run statistics
├── models/                All synthesised programs
│   └── model_NNNN.py
├── bugs/                  One JSON + reproducer per detected anomaly
│   ├── bug_crash_*.json
│   ├── bug_nan_*.json
│   ├── bug_inconsistent_*.json
│   ├── *.inputs.pt        Buggy input tensors
│   └── *.repro.py         Standalone reproducer
└── workspace/             Temporary subprocess files (safe to delete)
```

---

## Optional: offline state-sensitivity probe

By default SASFuzz uses heuristic σ and b tables from `core/state_signals.py`. To override with offline-probe results, place a `state_signals.json` file in the repo root:

```json
{
  "sigma": {
    "gradient_tracking":     {"torch.nn.CTCLoss": 1, "torch.linalg.svd": 1},
    "execution_mode":        {"torch.nn.BatchNorm1d": 1},
    "distribution_strategy": {"torch.nn.SyncBatchNorm": 1}
  },
  "bug_prior": {
    "gradient_tracking":     {"torch.nn.CTCLoss": 1},
    "execution_mode":        {"torch.nn.BatchNorm1d": 1},
    "distribution_strategy": {"torch.distributed.all_reduce": 1}
  }
}
```
