# SASFuzz

State-aware fuzzing of deep learning libraries via skeleton-guided synthesis.

## Setup

```bash
pip install torch==2.9.1 tensorflow==2.21.0 numpy requests openai
```

Set the API key for your chosen backend:

```bash
export OPENAI_API_KEY=your_key
# Qwen3.6-27B (RQ4) is served locally via Ollama — no API key required.
```

## LLM backends

The paper evaluates two backends (Section 4.2): GPT-5 (default) and Qwen3.6-27B
(open-source, RQ4 LLM sensitivity, served via Ollama).

| `--llm-backend` | Model | Key |
|---|---|---|
| `gpt5` *(default, paper)* | gpt-5 | `OPENAI_API_KEY` |
| `qwen` *(paper RQ4)* | qwen3.6:27b (Ollama) | — |

## Run

### PyTorch

```bash
python run_pytorch.py --mode full --output-dir results/pytorch
```

### TensorFlow

```bash
python run_tensorflow.py --out results/tensorflow
```

Stops automatically after 10 consecutive models introduce no new APIs.

---

## Reproducing paper results

### RQ1 — Empirical study of state-related issues (Tables 1 & 2)

RQ1 analyses 329 fix-verified correctness issues from PyTorch and TensorFlow to show that 62.6% are state-related and concentrate on three dimensions: gradient tracking, execution mode, and distribution strategy.

```bash
cd RQ1
python -m rq1.collect
python -m rq1.hydrate
python -m rq1.verify_fix
python -m rq1.classify
python -m rq1.report
```

### RQ2 & RQ3 — Coverage and bug detection (Tables 3 & 4)

The state-aware bug reproducer artifact is in `bugs/reproducers/state`. The
paper-counted artifact contains 47 standalone scripts: 18 PyTorch and 29
TensorFlow reproducers, distributed over gradient tracking (18), execution mode
(14), and distribution strategy (15). These scripts intentionally omit external
source metadata and maintainer outcome metadata.

Run one reproducer directly:

```bash
python bugs/reproducers/state/state_bug_001.py
```

**PyTorch:**
```bash
python scripts/evaluate.py --framework pytorch --models 1000 --budget 86400 \
    --out results/eval
```

**TensorFlow:**
```bash
python scripts/evaluate.py --framework tensorflow --models 1000 --budget 86400 \
    --out results/eval
```

**Both frameworks:**
```bash
python scripts/evaluate.py --framework both --models 1000 --budget 86400 \
    --out results/eval
```

To report from existing results without re-running:
```bash
python scripts/evaluate.py --framework both --out results/eval --report-only
```

### RQ4 — Ablation study and LLM sensitivity (Tables 5 & 6)


**Component ablation — PyTorch:**
```bash
python scripts/ablation.py --experiment ablation --framework pytorch \
    --models 1000 --budget 86400 --out results/ablation
```

**Component ablation — TensorFlow:**
```bash
python scripts/ablation.py --experiment ablation --framework tensorflow \
    --models 1000 --budget 86400 --out results/ablation
```

**LLM sensitivity — PyTorch:**
```bash
python scripts/ablation.py --experiment llm --framework pytorch \
    --models 1000 --budget 86400 --out results/ablation
```

**LLM sensitivity — TensorFlow:**
```bash
python scripts/ablation.py --experiment llm --framework tensorflow \
    --models 1000 --budget 86400 --out results/ablation
```

Ablation variants (`--ablation` flag):

| Variant | Meaning |
|---|---|
| `none` | Full SASFuzz |
| `no_skeleton` | LLM generates free-form programs (no state skeleton) |
| `no_scaffold` | Slot structure kept, state constructs moved to prompt only |
| `no_selection` | w/o State Rel. — sets σ = 0 in Eq. 1 (paper Table 5) |

---

## Output structure

```
results/<run>/
├── coverage.json       API coverage and run statistics
├── models/             All synthesised programs (model_NNNN.py)
└── bugs/               One JSON report + standalone reproducer per detected bug
```
