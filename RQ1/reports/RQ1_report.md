# RQ1 — Empirical Study: Live Numbers vs Paper Targets

All counts are recomputed from `RQ1/data/`. The 'Δ' column shows the gap vs the paper's reported number; `match` means they agree.

## Collection (raw → unique → filtered)

| Metric | Live | Paper | Δ |
|---|---:|---:|---:|
| Raw hits | 6011 | 6011 | match |
| Unique hits | 5897 | 5879 | Δ+18 |
| Filtered (correctness keywords) | 1122 | 1122 | match |
| Filtered — PyTorch | 622 | — | — |
| Filtered — TensorFlow | 500 | — | — |

### Per-query counts (raw search)

| Query | Repo | Labels | Count |
|---|---|---|---:|
| pytorch_autograd | pytorch/pytorch | triaged + module: autograd | 595 |
| pytorch_functorch | pytorch/pytorch | triaged + module: functorch | 667 |
| pytorch_dynamo | pytorch/pytorch | triaged + module: dynamo | 1819 |
| pytorch_ddp | pytorch/pytorch | triaged + module: ddp | 88 |
| pytorch_fsdp | pytorch/pytorch | triaged + module: fsdp | 382 |
| tensorflow_keras | tensorflow/tensorflow | type:bug + comp:keras | 857 |
| tensorflow_xla | tensorflow/tensorflow | type:bug + comp:xla | 182 |
| tensorflow_function | tensorflow/tensorflow | type:bug + comp:tf.function | 69 |
| tensorflow_dist_strat | tensorflow/tensorflow | type:bug + comp:dist-strat | 108 |
| tensorflow_ops | tensorflow/tensorflow | type:bug + comp:ops | 1244 |

## Fix-verification (Table 1)

| Repo | Fix-verified | Unverified | Excluded (PR) | Total |
|---|---:|---:|---:|---:|
| pytorch/pytorch | 194 | 428 | 0 | 622 |
| tensorflow/tensorflow | 35 | 465 | 0 | 500 |
| **Total** | **229** | **893** | **0** | **1122** |


| Comparison | Live | Paper | Δ |
|---|---:|---:|---:|
| Fix-verified total | 229 | 329 | Δ-100 |
| Unverified total | 893 | 793 | Δ+100 |
| Fix-verified PyTorch | 194 | 194 | match |
| Fix-verified TensorFlow | 35 | 135 | Δ-100 |

## Classification — backend = `heuristic` (Table 2)

_Note: 218 of 229 fix-verified issues were skipped because their body/labels are not yet hydrated. Run `rq1.hydrate --only-fix-verified` (with `GITHUB_TOKEN` for speed) to complete classification._

| Primary dimension | PyTorch | TF | Total |
|---|---:|---:|---:|
| A. gradient_tracking | 10 | 0 | 10 |
| B. execution_mode | 0 | 0 | 0 |
| C. distribution_strategy | 1 | 0 | 1 |
| D. other_state | 0 | 0 | 0 |
| E. non_state | 0 | 0 | 0 |
| **State-related (A–D)** | **11** | **0** | **11** |
| **Non-state (E)** | **0** | **0** | **0** |


| Comparison | Live | Paper | Δ |
|---|---:|---:|---:|
| State-related total | 11 | 206 | Δ-195 |
| A total | 10 | 55 | Δ-45 |
| A PyTorch | 10 | 41 | Δ-31 |
| A TensorFlow | 0 | 14 | Δ-14 |
| B total | 0 | 74 | Δ-74 |
| B PyTorch | 0 | 31 | Δ-31 |
| B TensorFlow | 0 | 43 | Δ-43 |
| C total | 1 | 62 | Δ-61 |
| C PyTorch | 1 | 43 | Δ-42 |
| C TensorFlow | 0 | 19 | Δ-19 |
| D total | 0 | 15 | Δ-15 |
| D PyTorch | 0 | 10 | Δ-10 |
| D TensorFlow | 0 | 5 | Δ-5 |

---

See `reports/DELTA_ANALYSIS.md` for the explanation of where live numbers diverge from the paper and what additional commands close the remaining gaps.

Reproducibility: `python -m rq1.collect` then `python -m rq1.hydrate` (or `--only-fix-verified`) then `python -m rq1.verify_fix` (or `--mode fast`) then `python -m rq1.classify --backend heuristic` then `python -m rq1.report`.
