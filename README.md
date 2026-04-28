# SASFuzz

State-aware fuzzing of deep learning libraries via skeleton-guided synthesis (PyTorch and TensorFlow).

## Setup

```bash
pip install torch>=2.9.0 tensorflow>=2.21.0 numpy requests openai
```

Set the API key for your chosen backend (DeepSeek is free at platform.deepseek.com):

```bash
export DEEPSEEK_API_KEY=your_key     # default backend — free
export OPENAI_API_KEY=your_key       # for --llm-backend gpt5
export ANTHROPIC_API_KEY=your_key    # for --llm-backend claude35
```

## LLM backends

| `--llm-backend` | Model | Key |
|---|---|---|
| `deepseek-v2` *(default)* | deepseek-chat | `DEEPSEEK_API_KEY` |
| `gpt5` | gpt-4o | `OPENAI_API_KEY` |
| `claude35` | claude-3-5-sonnet-20241022 | `ANTHROPIC_API_KEY` |

## Run

### PyTorch (1000 models, 60 s mutation budget each)

```bash
python main.py --mode full --output-dir results/pytorch
```

### TensorFlow (1000 models)

```bash
python run_tf.py --out results/tensorflow
```

### Both frameworks in parallel

```bash
python run_both.py
```

Stops automatically after 10 consecutive models introduce no new APIs.

---

## Reproducing paper results

### RQ1 — Empirical study of state-related issues (Tables 1 & 2)

RQ1 analyses 329 fix-verified correctness issues from PyTorch and TensorFlow to characterise how prevalent state-related bugs are and which state dimensions they concentrate on.

```bash
cd RQ1
python -m rq1.collect     # fetch issues from GitHub (needs GITHUB_TOKEN)
python -m rq1.hydrate     # download issue bodies
python -m rq1.classify    # LLM-assisted classification (needs API key)
python -m rq1.verify_fix  # filter to fix-verified issues only
python -m rq1.report      # print Tables 1 & 2
```

### RQ2 — API coverage and code coverage (Table 3)

Run SASFuzz for 1000 models on each framework and collect coverage metrics.

**PyTorch:**
```bash
python scripts/rq2_coverage.py --framework pytorch --models 1000 --budget 60 \
    --out results/rq2/pytorch
```

**TensorFlow:**
```bash
python scripts/rq2_coverage.py --framework tensorflow --models 1000 --budget 60 \
    --out results/rq2/tensorflow
```

Coverage numbers are written to `results/rq2/<framework>/coverage.json`. Baselines (Muffin, COMET, ModelMeta) must be run separately with their original artifacts.

### RQ3 — Bug detection (Table 4)

After running RQ2, aggregate all detected bugs:

**PyTorch:**
```bash
python scripts/rq3_bugs.py --results results/rq2/pytorch
```

**TensorFlow:**
```bash
python scripts/rq3_bugs.py --results results/rq2/tensorflow
```

**Both:**
```bash
python scripts/rq3_bugs.py --results results/rq2
```

### RQ4 — Ablation study and LLM sensitivity (Tables 5 & 6)

**Component ablation — PyTorch (Table 5):**
```bash
python scripts/rq4_ablation.py --experiment ablation --framework pytorch \
    --models 1000 --budget 60 --out results/rq4/pytorch
```

**Component ablation — TensorFlow (Table 5):**
```bash
python scripts/rq4_ablation.py --experiment ablation --framework tensorflow \
    --models 1000 --budget 60 --out results/rq4/tensorflow
```

**LLM sensitivity — PyTorch (Table 6):**
```bash
python scripts/rq4_ablation.py --experiment llm --framework pytorch \
    --models 1000 --budget 60 --out results/rq4/pytorch
```

**LLM sensitivity — TensorFlow (Table 6):**
```bash
python scripts/rq4_ablation.py --experiment llm --framework tensorflow \
    --models 1000 --budget 60 --out results/rq4/tensorflow
```

Ablation variants (`--ablation` flag):

| Variant | Meaning |
|---|---|
| `none` | Full SASFuzz |
| `no_skeleton` | LLM generates free-form programs (no state skeleton) |
| `no_scaffold` | Slot structure kept, state constructs moved to prompt only |
| `no_selection` | Uniform API selection (σ = b = 0 in Eq. 1) |
| `no_feedback` | Uniform mutation strategy sampling |

---

## Output structure

```
results/<run>/
├── coverage.json       API coverage and run statistics
├── models/             All synthesised programs (model_NNNN.py)
└── bugs/               One JSON report + standalone reproducer per detected bug
```
