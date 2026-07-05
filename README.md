# SASFuzz

State-aware fuzzing of deep learning libraries via skeleton-guided synthesis.

## Setup

```bash
pip install -r requirements.txt        # torch>=2.4, tensorflow>=2.15, numpy, requests, openai
```

A CUDA-capable GPU is required: the oracle is a CPU-vs-GPU differential test.

## LLM backends

| `--llm-backend` | Model | Key |
|---|---|---|
| `gpt5` *(default)* | gpt-5 (or any chat/o-series model via `--llm-model`) | `OPENAI_API_KEY` |
| `qwen` | Ollama model (default `qwen2.5-coder:32b`, override `--llm-model`) | — |

## Run


### PyTorch

```bash
python run_pytorch.py --mode full --llm-backend gpt5 --output-dir results/pytorch
```

### TensorFlow

```bash
python run_tensorflow.py --llm-backend gpt5 --out results/tensorflow
```

Stops automatically after 10 consecutive models introduce no new APIs.

---


### Empirical study of state-related issues

Empirical study analyses 329 fix-verified correctness issues from PyTorch and TensorFlow to show that 62.6% are state-related and concentrate on three dimensions: gradient tracking, execution mode, and distribution strategy.

```bash
cd empirical_study
python -m empirical_study.collect
python -m empirical_study.hydrate
python -m empirical_study.verify_fix
python -m empirical_study.classify
python -m empirical_study.report
```



---
