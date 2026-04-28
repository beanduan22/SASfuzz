# SASFuzz

State-aware fuzzing of DL libraries via skeleton-guided synthesis.

## Setup

```bash
pip install torch>=2.9.0 tensorflow>=2.21.0 numpy requests openai
export DEEPSEEK_API_KEY=your_key_here   # free at platform.deepseek.com
```

## Run

**PyTorch** (1000 models, 60 s mutation budget each):
```bash
python -m sasfuzz.main --mode full --output-dir results/pytorch
```

**TensorFlow**:
```bash
python sasfuzz/run_tf.py --out results/tensorflow
```

**Both in parallel**:
```bash
python sasfuzz/run_both.py
```

Stops automatically after 10 consecutive models introduce no new APIs.

## LLM backends

| `--llm-backend` | Model | Requires |
|---|---|---|
| `deepseek-v2` *(default)* | deepseek-chat | `DEEPSEEK_API_KEY` |
| `gpt5` | gpt-4o | `OPENAI_API_KEY` |
| `claude35` | claude-3-5-sonnet-20241022 | `ANTHROPIC_API_KEY` |
| `qwen25-32b` | Qwen2.5-Coder-32B | Ollama running locally |
| `ollama` | qwen2.5-coder:32b | Ollama running locally |

```bash
python -m sasfuzz.main --mode full --llm-backend claude35 --output-dir results/claude35
```

## RQ scripts

```bash
# RQ1 — empirical study
cd RQ1 && python -m rq1.report

# RQ2 — coverage
python scripts/rq2_coverage.py --framework both --models 100

# RQ3 — bug summary
python scripts/rq3_bugs.py --results results/

# RQ4 — ablation + LLM sensitivity
python scripts/rq4_ablation.py --experiment both --models 50
```

Ablation flag: `--ablation {none,no_skeleton,no_scaffold,no_selection,no_feedback}`

## Output

```
results/<run>/
├── coverage.json       API coverage and run statistics
├── models/             All synthesised programs
└── bugs/               JSON report + standalone reproducer per detected bug
```
