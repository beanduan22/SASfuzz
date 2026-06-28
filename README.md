# SASFuzz

State-aware fuzzing of deep learning libraries via skeleton-guided synthesis.

## Setup

```bash
pip install -r requirements.txt        # torch>=2.4, tensorflow>=2.15, numpy, requests, openai
```

A CUDA-capable GPU is required: the oracle is a CPU-vs-GPU differential test.

## How it works

Each of the 12 **state skeletons** (6 PyTorch + 6 TensorFlow) is a complete,
runnable program built from two parts:

* a **scaffold** with three typed slots the synthesizer fills — `LAYER_SLOT`,
  `BODY_SLOT`, `INPUT_SLOT` (the `Model` body and `make_inputs`);
* a **frozen `sas_run(device, inputs)` harness** that installs the target
  runtime state (autograd / `no_grad`, `train`/`eval`, `torch.jit.trace`,
  `DistributedDataParallel`, collectives; `GradientTape`, `tf.function`,
  `MirroredStrategy`, …) and returns the *state-dependent* outputs.

The executor runs `sas_run` on CPU and GPU and the oracle compares those
state outputs (plus crash / NaN / gradient checks), so every comparison is made
**under the installed runtime state** — never a bare forward pass.

## LLM backends

| `--llm-backend` | Model | Key |
|---|---|---|
| `gpt5` *(default)* | gpt-5 (or any chat/o-series model via `--llm-model`) | `OPENAI_API_KEY` |
| `qwen` | Ollama model (default `qwen2.5-coder:32b`, override `--llm-model`) | — |
| `template` | offline deterministic slot-filler | — |

```bash
export OPENAI_API_KEY=your_key          # only for --llm-backend gpt5
```

The **`template`** backend needs no API key or network: it fills the skeleton
slots with a small, shape-safe default body. Use it to verify the pipeline runs
end-to-end before spending tokens on an LLM backend.

## Run

### Quick smoke test (offline, no key, ~1 min)

```bash
python run_pytorch.py --mode subset --llm-backend template --output-dir results/pytorch
```

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

RQ1 analyses 329 fix-verified correctness issues from PyTorch and TensorFlow to show that 62.6% are state-related and concentrate on three dimensions: gradient tracking, execution mode, and distribution strategy.

```bash
cd RQ1
python -m rq1.collect
python -m rq1.hydrate
python -m rq1.verify_fix
python -m rq1.classify
python -m rq1.report
```

### Run Tool


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



---
